from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict ,Annotated
from dotenv import load_dotenv
from pydantic import BaseModel,Field
import operator

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')


class UpscEssayState(TypedDict):
    essay:str
    clarity_feedback:str
    depth_feedback:str
    language_feedback:str
    

    clarity_score:int
    depth_score:int
    langauage_score:int

    final_score:int
    final_feedback:str


class strFeedback(BaseModel):
    feedback:str = Field(description='Detail feedback for the essay ')
    score:int = Field(description='generate a score on the basis of feedback (0 to 10)',ge=0,le=10)

structred_output = model.with_structured_output(strFeedback)


def evaluate_clarity(state:UpscEssayState) -> UpscEssayState:

    essay = state['essay']

    prompt = f"evaluate the clarity of essay and provide a feedback on the basis of clarity and assign a score out of 10 \n{essay}"

    output = structred_output.invoke(prompt)

    return {'clarity_feedback':output.feedback,'clarity_score':int(output.score)}

def evaluate_depth(state:UpscEssayState) -> UpscEssayState:

    essay = state['essay']

    prompt = f"evaluate the depth of essay and provide a feedback on the basis of depth and assign a score out of 10 \n{essay}"

    output = structred_output.invoke(prompt)

    return {'depth_feedback':output.feedback,'depth_score':int(output.score)}

def evaluate_language(state:UpscEssayState) -> UpscEssayState:

    essay = state['essay']

    prompt = f"evaluate the language quality of essay and provide a feedback on the basis of language quality and assign a score out of 10 \n{essay}"

    output = structred_output.invoke(prompt)

    return {'language_feedback':output.feedback,'langauage_score':int(output.score)}

def evaluate_final(state:UpscEssayState) -> UpscEssayState:

    prompt = f"Generate a summarized feedback on the basis of \n language-feedback -{state['language_feedback']} \n depth-feedback - {state['depth_feedback']} \n clarity-feedback - {state['clarity_feedback']}"
    overall_feedback = model.invoke(prompt)
    state['final_feedback'] = overall_feedback
     
    clarity_score = state['clarity_score']
    depth_score = state['depth_score']
    language_score = state['langauage_score']

    average = (clarity_score + depth_score + language_score)/3


    
    state['final_score'] = average

    return state



graph = StateGraph(UpscEssayState)

graph.add_node('evaluate_clarity',evaluate_clarity)
graph.add_node('evaluate_depth',evaluate_depth)
graph.add_node('evaluate_language',evaluate_language)

graph.add_node('evaluate_final',evaluate_final)

graph.add_edge(START,'evaluate_clarity')
graph.add_edge(START,'evaluate_depth')
graph.add_edge(START,'evaluate_language')

graph.add_edge('evaluate_clarity','evaluate_final')
graph.add_edge('evaluate_depth','evaluate_final')
graph.add_edge('evaluate_language','evaluate_final')

graph.add_edge('evaluate_final',END)

workflow = graph.compile()

essay = """
Terrorism is one of the biggest threats faced by the modern world. It refers to the use of violence, fear, and intimidation by individuals or groups to achieve political, religious, or ideological goals. Terrorists attack innocent people, destroy public property, and create an atmosphere of fear in society. Their main aim is to disturb peace, destabilize governments, and draw attention to their demands.

There are many causes of terrorism, such as political unrest, religious extremism, social injustice, poverty, and foreign influence. Sometimes young people are misguided and brainwashed into joining terrorist organizations. Terrorism affects the lives of common people by taking away their sense of safety and causing loss of life and property.

To fight terrorism, countries must work together and share information. Governments should strengthen security, improve intelligence systems, and take steps to stop the spread of extremist ideas. Education, employment, and awareness programs can also help prevent the youth from being influenced by terrorism. Citizens must stay alert and cooperate with security agencies.

In conclusion, terrorism is a global problem that requires global cooperation. Peace, unity, and understanding are the best weapons to defeat terrorism. Only by working together can we build a safe and peaceful world for future generations.

"""

inital_state = {
    'essay':essay
    }

result = workflow.invoke(inital_state)

print(f"Result : ", result)


