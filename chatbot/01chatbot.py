from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from dotenv import load_dotenv

load_dotenv()

class messageState(TypedDict):
    messages: Annotated[list[str],add_messages]

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

def chat_node(state:messageState) -> messageState:

    message = state['messages']

    response = llm.invoke(message)

    return {'messages':[response]}


checkpointer = InMemorySaver()
graph = StateGraph(messageState)



graph.add_node('chat_node',chat_node)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile(checkpointer=checkpointer)

thread_id = 1

while True:
    Usermessage = input("Type the message here : ")

    print(f"User : " , Usermessage)

    if Usermessage.strip().lower() in ['bye','exit','quit']:
        break

    config = {
        'configurable':{
            'thread_id':thread_id
        }
    }

    message = {
        'messages':[Usermessage]
    }

    result = chatbot.invoke(message,config=config)
    print(f"Result : ",result['messages'][-1].content)
