from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Literal, TypedDict
from dotenv import load_dotenv

load_dotenv()

class rootState(TypedDict):
    A:int
    B:int
    C:int
    equation:str
    discriminant : float
    result : str
   

def show_equation(state:rootState) -> rootState:
    equation = f"{state['A']}x^2+{state['B']}x+{state['C']}"
    return {'equation':equation}

def calculate_discriminant(state:rootState) ->rootState:
    b = state['B']
    a = state['A']
    c = state['C']

    discriminant = (b*b) - (4*a*c)
    return {'discriminant':discriminant}

def real_root(state:rootState) -> rootState:
    b = state['B']
    d = state['discriminant']
    a = state['A']

    root1 = (-b+(d**0.5))/(2*a)
    root2 = (-b-(d**0.5))/(2*a)

    result = f"Root 1 : {root1} and Root 2 : {root2}"

    return {'result':result}

def repeated_root(state:rootState) -> rootState:
    b = state['B']
    a = state['A']

    root = (-b)/(2*a)

    result = f"Root  : {root} "

    return {'result':result}

def no_real_root(state:rootState) -> rootState:
    result  = f"No real roots exist"
    return {'result' : result}

def check_condition(state:rootState) -> Literal["real_root","repeated_root","no_real_root"]:

    if state['discriminant'] > 0:
        return "real_root"
    elif state['discriminant'] == 0:
        return "repeated_root"
    else :
        return "no_real_root"
    

  
  


graph = StateGraph(rootState)

graph.add_node('show_equation',show_equation)
graph.add_node('calculate_discriminant',calculate_discriminant)
graph.add_node('real_root',real_root)
graph.add_node('repeated_root',repeated_root)
graph.add_node('no_real_root',no_real_root)

graph.add_edge(START,'show_equation')
graph.add_edge('show_equation','calculate_discriminant')
graph.add_conditional_edges('calculate_discriminant',check_condition)

graph.add_edge('real_root',END)
graph.add_edge('repeated_root',END)
graph.add_edge('no_real_root',END)

workflow = graph.compile()

initial_state = {
    "A":10,
    "B":2,
    "C":1
}

result  = workflow.invoke(initial_state)
print(f"Result : ",result)

