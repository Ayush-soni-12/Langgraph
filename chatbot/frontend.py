import streamlit as st
from chatbot import chatbot


config = {
    'configurable':{
        'thread_id':2
    }
}



st.title("GenAI Chat UI")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask something...")

if user_input:
    with st.chat_message("user"):
        st.text(user_input)


    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            result = chatbot.invoke({'messages':[user_input]},config=config)
            response = result['messages'][-1].content

            st.text(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    st.rerun()
