import stat
from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

model  = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class blogState(TypedDict):
    title:str
    outline:str
    content:str

def createOutline(state:blogState) -> blogState:
    title = state['title']
    prompt = f"generate a outline for this title {title}"
    result = model.invoke(prompt)
    state['outline'] = result.content
    return state

def generateBlog(state:blogState) -> blogState:
    title = state['title']
    outline = state['outline']

    prompt = f"Generate a blog for the title {title} using the this outline \n {outline}"

    result = model.invoke(prompt)
    state['content'] = result.content

graph = StateGraph(blogState)

graph.add_node('create_outline',createOutline)
graph.add_node('generate_blog',generateBlog)

graph.add_edge(START,'create_outline')
graph.add_edge('create_outline','generate_blog')
graph.add_edge('generate_blog',END)

workflow = graph.compile()

initial_state = {"title":"machine learning"}

final_state = workflow.invoke(initial_state)

print(f"Blog " , final_state)