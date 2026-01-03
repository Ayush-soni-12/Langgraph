import os
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool

# Load environment variables
load_dotenv()

# Database Configuration
DB_URI = "postgresql://Ayush:Ayush123@localhost:5432/langgraph"
STOCK_API_KEY = os.getenv("STOCK_API")

# Create a persistent connection pool
# We use this pool in chatbot.py for the checkpointer
pool = ConnectionPool(conninfo=DB_URI, max_size=20)