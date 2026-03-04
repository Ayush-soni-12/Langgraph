from itertools import count
import requests
from typing import Annotated, TypedDict
import uuid

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.messages.utils import trim_messages, count_tokens_approximately
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

# FIX 1: Import store
from init_db import pool, STOCK_API_KEY, store

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


def make_memory_tools(user_id: str):
    @tool
    def save_memory(key: str, value: str) -> str:
        """Save an important fact about the user for future conversations."""
        store.put(
            ("user_memory", user_id),
            key,
            {"data": value}
        )
        return f"Saved: {key} = {value}"

    @tool
    def get_memory(key: str = None) -> str:
        """Retrieve facts about the user from long-term memory."""
        if key:
            result = store.get(("user_memory", user_id), key)
            return f"{key}: {result.value['data']}" if result else f"No memory for: {key}"
        results = store.search(("user_memory", user_id))
        if not results:
            return "No memories found."
        return "\n".join([f"{r.key}: {r.value['data']}" for r in results])

    return [save_memory, get_memory]


# FIX 2: Only static tools here — memory tools are added dynamically
static_tools = [search_tool, get_stock_price, calculator]

# ======================================================
# 2. Model & Graph Setup
# ======================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_output_tokens=500,
    temperature=0.7
)

class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str


def summarize_messages(state: MessageState):
    summary = state.get("summary", "")
    messages = state["messages"]

    if len(messages) > 10:
        to_summarize = messages[:-5]
        summary_prompt = (
            f"Extend the current summary by incorporating the new messages below: {summary}\n\n"
            f"New messages to summarize: {to_summarize}"
        )
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        print("Response summary", response.content)
        return {"summary": response.content}

    return {"summary": summary}


def chat_node(state: MessageState, config: RunnableConfig) -> MessageState:
    summary = state.get("summary", "")

    user_id = config["configurable"].get("user_id", "default_user")

    # FIX 3: Build all tools dynamically with correct user_id
    memory_tools = make_memory_tools(user_id)
    all_tools = static_tools + memory_tools

    # FIX 4: Bind the complete tool list — including memory tools
    llm_with_tools = llm.bind_tools(all_tools)

    # Load long-term memories

    memories = store.search(("user_memory", user_id)) 
    memory_text = ""
    if memories:
        facts = "\n".join([f"- {m.key}: {m.value['data']}" for m in memories])
        memory_text = f"\nWhat you know about this user:\n{facts}"

    recent_messages = trim_messages(
        state['messages'],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=1000,
        start_on="human",
        include_system=True
    )

    inputs = [
        SystemMessage(content=(
            f"You are a helpful assistant.{memory_text}"
            + (f"\n\nSummary of recent conversation: {summary}" if summary else "")
        ))
    ]
    inputs.extend(recent_messages)

    response = llm_with_tools.invoke(inputs)
    return {"messages": [response]}


# FIX 5: Custom tool node that also builds memory tools dynamically
def dynamic_tool_node(state: MessageState, config: RunnableConfig):
    user_id = config["configurable"].get("user_id", "default_user")
    memory_tools = make_memory_tools(user_id)
    all_tools = static_tools + memory_tools
    return ToolNode(all_tools).invoke(state, config)  # ← Correct tools every time


builder = StateGraph(MessageState)
builder.add_node("summarize", summarize_messages)
builder.add_node("chat_node", chat_node)
builder.add_node("tools", dynamic_tool_node)  # ← Use dynamic node

builder.add_edge(START, "summarize")
builder.add_edge("summarize", "chat_node")
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

# FIX 6: No st import needed — accept user_id as a parameter
def get_config(thread_id: str, user_id: str = "default_user"):
    return {"configurable": {"thread_id": thread_id, "user_id": user_id}}

def format_msg(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                text_parts.append(item["text"])
        return "".join(text_parts)
    return ""

def load_messages_from_langgraph(thread_id, user_id="default_user"):
    if thread_id:
        state = chatbot.get_state(config=get_config(thread_id, user_id))
    if state:
        messages = state.values.get("messages", []) if state.values else []

    ui_messages = []
    for m in messages:
        if isinstance(m, ToolMessage):
            continue
        if isinstance(m, AIMessage) and m.tool_calls and not m.content:
            continue
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        clean_content = format_msg(m.content)
        if clean_content:
            ui_messages.append({"role": role, "content": clean_content})

    return ui_messages

def get_all_thread_ids():
    config_list = checkpointer.list(None)
    thread_ids = set()
    for item in config_list:
        thread_ids.add(item.config["configurable"]["thread_id"])
    return list(thread_ids)

def get_thread_title(thread_id, user_id="default_user"):
    state = chatbot.get_state(config=get_config(thread_id, user_id))
    messages = state.values.get("messages", [])
    for m in messages:
        if isinstance(m, HumanMessage):
            return (m.content[:20] + "...") if len(m.content) > 20 else m.content
    return "New Chat"