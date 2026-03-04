import os
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from psycopg import Connection
from langgraph.store.postgres import PostgresStore

load_dotenv()

DB_URI = "postgresql://Ayush:Ayush123@localhost:5432/Genai"
STOCK_API_KEY = os.getenv("STOCK_API")

# For checkpointer (existing)
pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=20,
    kwargs={
        "autocommit": True,
        "row_factory": dict_row
    }
)

# For long-term memory store (new)
store_conn = Connection.connect(
    DB_URI,
    autocommit=True,
    prepare_threshold=0
)
store = PostgresStore(store_conn)
store.setup()