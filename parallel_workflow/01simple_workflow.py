from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

class BatsmanState(TypedDict):
    runs:int
    ball:int
    four:int
    six:int

    sr:float
    bpb:float
    boundary_percentage:float
    summary:str

def caluclate_sr(state:BatsmanState)->BatsmanState:

    sr = (state['runs']/state['ball'])*100
    return {'sr':sr}

def calculate_bpb(state:BatsmanState) -> BatsmanState:
    bpb = state['ball']/(state['four'] + state['six'])
    return {'bpb':bpb}

def calculate_bp(state:BatsmanState) -> BatsmanState:
    boundary_percentage = (((state['four']*4) + (state['six']*6))/state['runs']) * 100
    return {'boundary_percentage':boundary_percentage}

def summary(state:BatsmanState) -> BatsmanState:

    summary = f"""
      Strike rate {state['sr']} \n
      Ball per boundary {state['bpb']} \n
      boundary percentage {state['boundary_percentage']} \n
"""
    
    state['summary'] = summary
    return state

graph = StateGraph(BatsmanState)

graph.add_node('calculate_sr',caluclate_sr)
graph.add_node('calculate_bpb',calculate_bpb)
graph.add_node('calculate_bp',calculate_bp)
graph.add_node('summary',summary)


graph.add_edge(START,'calculate_sr')
graph.add_edge(START,'calculate_bpb')
graph.add_edge(START,'calculate_bp')

graph.add_edge('calculate_sr','summary')
graph.add_edge('calculate_bpb','summary')
graph.add_edge('calculate_bp','summary')

graph.add_edge('summary',END)

workflow = graph.compile()

initial_state ={
    'runs':100,
    'ball':45,
    'four':10,
    'six':6
}

final_state = workflow.invoke(initial_state)
print(final_state['summary'])


