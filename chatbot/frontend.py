import streamlit as st
from chatbot import chatbot
from langchain.messages import HumanMessage
import uuid

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

if "threads" not in st.session_state:
    # thread_id -> title
    st.session_state.threads = {}

if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False


# ======================================================
# Sidebar
# ======================================================

st.sidebar.title("GenAI Chat")

# ---- New Chat Button ----
if st.sidebar.button(
    "New Chat",
    key="new_chat_btn",
    disabled=st.session_state.is_streaming
):
    # Only create a new chat if current one has messages
    current_msgs = load_messages_from_langgraph(st.session_state.thread_id)
    if current_msgs:
        st.session_state.thread_id = generate_thread_id()

st.sidebar.subheader("Chats")

# ---- Chat List ----
for tid, title in reversed(st.session_state.threads.items()):
    if st.sidebar.button(
        title,
        key=f"chat_btn_{tid}",
        disabled=st.session_state.is_streaming
    ):
        st.session_state.thread_id = tid


# ======================================================
# Render Messages (ALWAYS from LangGraph)
# ======================================================

messages = load_messages_from_langgraph(st.session_state.thread_id)

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ======================================================
# Chat Input
# ======================================================

user_input = st.chat_input(
    "Ask something...",
    disabled=st.session_state.is_streaming
)

# 🚨 HARD GUARD (THIS FIXES YOUR ISSUE)
if st.session_state.is_streaming:
    st.stop()

if user_input:
    st.session_state.is_streaming = True

    # Register chat title on FIRST user message
    if st.session_state.thread_id not in st.session_state.threads:
        st.session_state.threads[
            st.session_state.thread_id
        ] = user_input[:20]

    # Show user message instantly
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        with st.spinner("Thinking..."):
            result = chatbot.stream(
                {"messages": [user_input]},
                config=get_config(st.session_state.thread_id),
                stream_mode="messages"
            )

            for chunk, _ in result:
                if chunk.content:
                    full_response += chunk.content
                    placeholder.markdown(full_response)

    st.session_state.is_streaming = False
    st.rerun()
