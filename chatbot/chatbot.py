from itertools import count
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
from langchain_core.messages.utils import trim_messages,count_tokens_approximately
from langchain_core.messages import SystemMessage

# Local Imports
from init_db import pool, STOCK_API_KEY


# ======================================================
# Rag implementation
# ======================================================






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
    summary:str


MAX_TOKEN=1000


def summarize_messages(state: MessageState):
    summary = state.get("summary", "")
    messages = state["messages"]
    
    # Only summarize if we have more than, say, 10 messages
    if len(messages) > 10:
        # We summarize everything EXCEPT the last 5 messages 
        # (to keep the current conversation flow perfectly clear)
        to_summarize = messages[:-5]
        
        summary_prompt = (
            f"Extend the current summary by incorporating the new messages below: {summary}\n\n"
            f"New messages to summarize: {to_summarize}"
        )
        
        # Call LLM to create the summary
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        print(f"Response summary",response.content)
        
        # We return the NEW summary. 
        # IMPORTANT: We do NOT delete messages here so they stay in UI.
        return {"summary": response.content}
    
    return {"summary": summary}


def chat_node(state: MessageState) -> MessageState:
    summary = state.get("summary", "")
    
    # VIRTUAL TRIM: Only grab the most recent messages for the LLM
    # This does NOT delete them from Postgres/State.
    recent_messages = trim_messages(
        state['messages'],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=1000, 
        start_on="human",
        include_system=True
    )

    # Combine Summary (as a System Message) + Recent Messages
    inputs = []
    if summary:
        inputs.append(SystemMessage(content=f"Summary of previous conversation: {summary}"))
    
    inputs.extend(recent_messages)

    response = llm_with_tools.invoke(inputs)
    return {"messages": [response]}

tool_node = ToolNode(tools)

builder = StateGraph(MessageState)
builder.add_node("summarize", summarize_messages) # New Node
builder.add_node("chat_node", chat_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "summarize") # Summarize first
builder.add_edge("summarize", "chat_node") # Then chat
builder.add_conditional_edges("chat_node", tools_condition)
builder.add_edge("tools", "chat_node")
builder.add_edge("chat_node", END)

checkpointer = PostgresSaver(pool)

checkpointer.setup()

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
            elif isinstance(item, dict):
                if "text" in item:
                    text_parts.append(item["text"])
        return "".join(text_parts)
    return ""

def load_messages_from_langgraph(thread_id):
    """
    Fetch history from DB, but CLEAN it before returning.
    Removes ToolMessages and formats messy text.
    """
    if thread_id:
        state = chatbot.get_state(config=get_config(thread_id)) 
    # print("State",state)
    if state:
        messages = state.values.get("messages", []) if state.values else []

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