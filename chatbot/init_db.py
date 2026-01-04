import os
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row

# Load environment variables
load_dotenv()

# Database Configuration
DB_URI = "postgresql://Ayush:Ayush123@localhost:5432/Genai"
STOCK_API_KEY = os.getenv("STOCK_API")

# Create a persistent connection pool
# We use this pool in chatbot.py for the checkpointer
pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=20,
    kwargs={
        "autocommit": True,       # <--- THIS FIXES THE "INDEX" ERROR
        "row_factory": dict_row   # <--- REQUIRED BY LANGGRAPH TO READ DATA
    }
)