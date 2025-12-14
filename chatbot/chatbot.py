import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List
from psycopg_pool import ConnectionPool # Required for persistent connection

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

# 1. Setup Database Connection Pool
DB_URI = "postgresql://Ayush:Ayush%40123@localhost:5432/langgraph"

# We use a connection pool so the connection stays open for the Streamlit app
pool = ConnectionPool(conninfo=DB_URI, max_size=20)

# 2. State & LLM
class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Note: Changed to 1.5-flash as 2.5 is not standard/public yet. 
# Change back if you have specific access.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") 

# 3. Nodes
def chat_node(state: MessageState) -> MessageState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 4. Graph Construction
builder = StateGraph(MessageState)
builder.add_node("chat_node", chat_node)
builder.add_edge(START, "chat_node")
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