from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.store.memory import InMemoryStore
from dotenv import load_dotenv ,find_dotenv
from pathlib import Path
import os

# Go to project root → then load .env

load_dotenv(find_dotenv())

google_api_key = os.getenv("GEMINI_API_KEY")
print(google_api_key)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


store = InMemoryStore(index={'embed': embedding_model,'dims':768})

namespace = ("user", "u1")

# Add 10 memories with keys 1..10 and different data
memories = [
    "user likes pizza",
    "user prefers dark mode",
    "user drinks coffee in the morning",
    "user works out on weekends",
    "user uses Linux for development",
    "user enjoys sci-fi movies",
    "user listens to lo-fi music while coding",
    "user prefers concise answers",
    "user is learning LangGraph",
    "user usually sleeps before midnight",
]

for key, memory in enumerate(memories, start=1):
    store.put(namespace, key, {"data": memory})

 

items=store.search(namespace,query = "user like dark mode",limit=1)

for item in items:
    print(item.value)