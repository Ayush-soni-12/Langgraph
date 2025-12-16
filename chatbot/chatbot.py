import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List
from psycopg_pool import ConnectionPool # Required for persistent connection
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode , tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests

load_dotenv()
stock_api = os.getenv("STOCK_API")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") 


# llm1 = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-7B-Instruct-v0.2",
#     task="text-generation",

# )

# chat_model = ChatHuggingFace(llm=llm1)


# Search tool

search_tool = DuckDuckGoSearchRun(region="us-en")


def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}
    

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={stock_api}"
    r = requests.get(url)
    return r.json()



tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)


# 1. Setup Database Connection Pool
DB_URI = "postgresql://Ayush:Ayush%40123@localhost:5432/langgraph"

# We use a connection pool so the connection stays open for the Streamlit app
pool = ConnectionPool(conninfo=DB_URI, max_size=20)

# 2. State & LLM
class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Note: Changed to 1.5-flash as 2.5 is not standard/public yet. 
# Change back if you have specific access.


# 3. Nodes
def chat_node(state: MessageState) -> MessageState:
    """LLm node that may answer or request a tool call """
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

# 4. Graph Construction
builder = StateGraph(MessageState)
builder.add_node("chat_node", chat_node)
builder.add_node("tools",tool_node)
builder.add_edge(START, "chat_node")
builder.add_conditional_edges("chat_node",tools_condition)
builder.add_edge("tools","chat_node")
builder.add_edge("chat_node", END)

# 5. Compile with Checkpointer
checkpointer = PostgresSaver(pool)
checkpointer.setup() # Ensures tables are created in DB

chatbot = builder.compile(checkpointer=checkpointer)

# ======================================================
# Database Helper Functions
# ======================================================

def get_all_thread_ids():
    """Retrieves all unique thread_ids from the checkpoints."""
    # PostgresSaver.list returns an iterator of CheckpointTuples
    # We pass None to list all threads
    config_list = checkpointer.list(None)
    
    thread_ids = set()
    for item in config_list:
        thread_ids.add(item.config["configurable"]["thread_id"])
    
    return list(thread_ids)

def get_thread_title(thread_id):
    """
    Fetches the first user message of a thread to use as a title.
    Returns 'New Chat' if empty.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)
    messages = state.values.get("messages", [])
    
    if messages:
        # Find the first human message
        for m in messages:
            if isinstance(m, HumanMessage):
                return len(m.content) > 20 and m.content[:20] + "..." or m.content
                
    return "New Chat"