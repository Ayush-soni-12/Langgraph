import streamlit as st
from chatbot import chatbot

config = {
    "configurable": {
        "thread_id": 2
    }
}

st.title("GenAI Chat UI")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask something...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.write(user_input)

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    # Assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        with st.spinner("Thinking..."):
            result = chatbot.stream(
                {"messages": [user_input]},
                config=config,
                stream_mode="messages"
            )

            for message_chunk, metadata in result:
                if message_chunk.content:
                    full_response += message_chunk.content
                    placeholder.markdown(full_response)

    # Save final assistant message ONCE
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

    st.rerun()
