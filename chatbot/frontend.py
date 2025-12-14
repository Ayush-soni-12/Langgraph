import streamlit as st
import uuid
from langchain_core.messages import HumanMessage
from chatbot import chatbot, get_all_thread_ids, get_thread_title

st.set_page_config(page_title="GenAI Chat UI", layout="wide")
st.title("GenAI Chat UI")

# ======================================================
# Helpers
# ======================================================

def generate_thread_id():
    return str(uuid.uuid4())

def get_config(thread_id):
    return {
        "configurable": {
            "thread_id": thread_id
        }
    }

def load_messages_from_langgraph(thread_id):
    """Fetch current state messages from DB"""
    state = chatbot.get_state(config=get_config(thread_id))
    messages = state.values.get("messages", [])

    ui_messages = []
    for m in messages:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        ui_messages.append({
            "role": role,
            "content": m.content
        })
    return ui_messages

# ======================================================
# Session State Init
# ======================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False

# ======================================================
# Sidebar - Database Integration
# ======================================================

st.sidebar.title("History")

# 1. New Chat Button
if st.sidebar.button("➕ New Chat", use_container_width=True):
    st.session_state.thread_id = generate_thread_id()
    st.rerun()

st.sidebar.markdown("---")

# 2. Fetch all threads from Postgres
db_threads = get_all_thread_ids()

# 3. Display Threads
# We reverse them roughly to show newest (based on ID extraction) or just list them.
# Note: In a real prod app, you'd sort by 'updated_at' timestamp.
if not db_threads:
    st.sidebar.info("No history found.")
else:
    for tid in db_threads[::-1]:
        # Get title dynamically from the DB history
        title = get_thread_title(tid)
        
        # Highlight current chat
        if tid == st.session_state.thread_id:
            if st.sidebar.button(f"🔵 {title}", key=tid, use_container_width=True):
                st.session_state.thread_id = tid
                st.rerun()
        else:
            if st.sidebar.button(f"{title}", key=tid, use_container_width=True):
                st.session_state.thread_id = tid
                st.rerun()

# ======================================================
# Main Chat Logic
# ======================================================

# Load history for the CURRENT thread_id
messages = load_messages_from_langgraph(st.session_state.thread_id)

# Render Messages
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
user_input = st.chat_input("Ask something...", disabled=st.session_state.is_streaming)

if user_input:
    st.session_state.is_streaming = True
    
    # Show user message instantly
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream Response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        with st.spinner("Thinking..."):
            # Stream events from LangGraph
            # stream_mode="messages" yields (chunk, metadata)
            result = chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=get_config(st.session_state.thread_id),
                stream_mode="messages"
            )

            for chunk, metadata in result:
                # Depending on the graph structure, the chunk might be an AIMessageChunk
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    placeholder.markdown(full_response)

    st.session_state.is_streaming = False
    st.rerun()