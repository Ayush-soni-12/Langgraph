import requests
from typing import Annotated, TypedDict, List
import uuid

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# Local Imports
from init_db import pool, STOCK_API_KEY

# ======================================================
# 1. Tool Definitions
# ======================================================

search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation (add, sub, mul, div)."""
    try:
        if operation == "add": result = first_num + second_num
        elif operation == "sub": result = first_num - second_num
        elif operation == "mul": result = first_num * second_num
        elif operation == "div":
            if second_num == 0: return {"error": "Division by zero"}
            result = first_num / second_num
        else: return {"error": f"Unsupported operation '{operation}'"}
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a symbol (e.g. 'AAPL') via Alpha Vantage."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={STOCK_API_KEY}"
    r = requests.get(url)
    return r.json()

tools = [search_tool, get_stock_price, calculator]

# ======================================================
# 2. Model & Graph Setup
# ======================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_output_tokens=500,  # <--- SET LIMIT HERE (e.g., 500 words/tokens)
    temperature=0.7         # Optional: Controls creativity (0.0 = Precise, 1.0 = Creative)
)
llm_with_tools = llm.bind_tools(tools)

class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]




def chat_node(state: MessageState) -> MessageState:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

builder = StateGraph(MessageState)
builder.add_node("chat_node", chat_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "chat_node")
builder.add_conditional_edges("chat_node", tools_condition)
builder.add_edge("tools", "chat_node")
builder.add_edge("chat_node", END)

checkpointer = PostgresSaver(pool)
chatbot = builder.compile(checkpointer=checkpointer)

# ======================================================
# 3. Helper Functions (Used by UI)
# ======================================================

def generate_thread_id():
    return str(uuid.uuid4())

def get_config(thread_id):
    return {"configurable": {"thread_id": thread_id}}

def format_msg(content):
    """
    CLEANER FUNCTION:
    - Handles standard strings.
    - Handles Gemini's list of dicts [{'type': 'text', 'text': ...}].
    """
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Extract text from the messy list
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
        return "".join(text_parts)
    return ""

def load_messages_from_langgraph(thread_id):
    """
    Fetch history from DB, but CLEAN it before returning.
    Removes ToolMessages and formats messy text.
    """
    state = chatbot.get_state(config=get_config(thread_id))
    messages = state.values.get("messages", [])

    ui_messages = []
    for m in messages:
        # 1. Skip technical ToolMessages
        if isinstance(m, ToolMessage):
            continue
        
        # 2. Skip empty tool calls
        if isinstance(m, AIMessage) and m.tool_calls and not m.content:
            continue

        role = "user" if isinstance(m, HumanMessage) else "assistant"
        
        # 3. Clean the content
        clean_content = format_msg(m.content)
        
        if clean_content:
            ui_messages.append({"role": role, "content": clean_content})
            
    return ui_messages

def get_all_thread_ids():
    """List all threads from Postgres."""
    config_list = checkpointer.list(None)
    thread_ids = set()
    for item in config_list:
        thread_ids.add(item.config["configurable"]["thread_id"])
    return list(thread_ids)

def get_thread_title(thread_id):
    """Get a simple title based on the first user message."""
    state = chatbot.get_state(config=get_config(thread_id))
    messages = state.values.get("messages", [])
    for m in messages:
        if isinstance(m, HumanMessage):
            return (m.content[:20] + "...") if len(m.content) > 20 else m.content
    return "New Chat"