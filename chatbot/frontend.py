import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Import from our logic file
from chatbot import (
    chatbot, 
    get_config, 
    generate_thread_id, 
    load_messages_from_langgraph, 
    get_all_thread_ids, 
    get_thread_title, 
    format_msg
)

st.set_page_config(page_title="GenAI Chat UI", layout="wide")

# ======================================================
# 1. Session State Initialization
# ======================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()
    # Initial Load
    st.session_state.messages = load_messages_from_langgraph(st.session_state.thread_id)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False

# ======================================================
# 2. Sidebar (History Management)
# ======================================================

st.sidebar.title("GenAI History")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    st.session_state.thread_id = generate_thread_id()
    st.session_state.messages = [] 
    st.rerun()

st.sidebar.markdown("---")

db_threads = get_all_thread_ids()

if db_threads:
    # Show newest first
    for tid in db_threads[::-1]:
        title = get_thread_title(tid)
        is_active = (tid == st.session_state.thread_id)
        
        label = f"🔵 {title}" if is_active else f"{title}"
        
        if st.sidebar.button(label, key=tid, use_container_width=True):
            if st.session_state.thread_id != tid:
                st.session_state.thread_id = tid
                # OPTIMIZATION: Only fetch from DB when switching threads
                st.session_state.messages = load_messages_from_langgraph(tid)
                st.rerun()

# ======================================================
# 3. Main Chat Interface
# ======================================================

st.title("GenAI Assistant")

# A. Render Chat from Local Session State (Fast!)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# B. Input Handler
user_input = st.chat_input("Ask something...", disabled=st.session_state.is_streaming)

if user_input:
    st.session_state.is_streaming = True
    
    # 1. Display User Message & Save to Local State
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Stream Assistant Response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        with st.spinner("Thinking..."):
            result = chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=get_config(st.session_state.thread_id),
                stream_mode="messages"
            )
            
            for chunk, metadata in result:
                # Filter out raw Tool outputs
                if isinstance(chunk, ToolMessage):
                    continue
                
                # Process AI text
                if isinstance(chunk, AIMessage):
                    if chunk.tool_calls and not chunk.content:
                        continue
                        
                    # Use helper from chatbot.py to clean text
                    chunk_text = format_msg(chunk.content)
                    full_response += chunk_text
                    
                    if full_response:
                        placeholder.markdown(full_response)
        
        # 3. Save Final Response to Local State
        if full_response:
            st.session_state.messages.append({"role": "assistant", "content": full_response})

    st.session_state.is_streaming = False