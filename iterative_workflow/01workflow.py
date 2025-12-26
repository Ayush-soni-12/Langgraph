from httpx import post
from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Literal, NotRequired, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.messages import SystemMessage , HumanMessage

load_dotenv()

class postState(TypedDict):
    topic:str
    tweet:str
    evaulation:Literal["approved","need_improvement"]
    feedback:str
    iteration:NotRequired[int] 
    max_iteration:NotRequired[int]

generator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
evaluator_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
optimizer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class evaluate_tweet(BaseModel):
    evaluation: Literal["approved","need_improvement"] = Field(...,description="give the final evaluation")
    feedback: str = Field(...,description="give the feedback for the tweet")

structred_model = evaluator_llm.with_structured_output(evaluate_tweet)


def generate_tweet(state:postState) -> postState:
    message = [
        SystemMessage(content="""You are a viral Twitter/X creator known for sharp, culturally-aware humor that feels both authentic and shareable. Your style combines:
    - Observational wit with unexpected twists
    - Meme-native thinking and internet-savvy references
    - Punchy, conversational delivery that feels like a friend's clever take
    - Relatable humor that taps into universal experiences with specific details
    - Strategic use of irony, understatement, or playful exaggeration"""),

        HumanMessage(content=f"""Craft a viral-ready tweet on: "{state['topic']}"

    **Core Strategy:**
    1. **Hook Immediately**: First line must intrigue or establish a relatable premise
    2. **Authentic Voice**: Sound like a real person, not a brand or AI
    3. **Cultural Resonance**: Optionally reference current internet trends/behaviors if relevant
    4. **Economical Wit**: Every word serves the joke or insight
    5. **Shareability Factor**: Create that "this is so true" or "need to send this to..." feeling

    **Creative Constraints:**
    - ABSOLUTELY NO question-answer format or rhetorical setups
    - Maximum 260 characters (leaves room for retweet comments)
    - Use natural, conversational English with occasional slang if it fits
    - Favor specific scenarios over vague statements
    - End strong: last line should deliver the punchline or insight

    **Advanced Techniques (choose one or blend):**
    - **Incongruity Juxtaposition**: Pair unexpected elements
    - **Rule of Three**: Setup, pattern, subversion
    - **Hyper-Specific Relatability**: "You know when..." with oddly precise details
    - **Meme Logic**: Apply internet thinking to real-world situations
    - **Quiet Observational**: Understated truth that hits harder because it's subtle

    **Avoid:**
    - Explaining the joke
    - Overused formats ("Nobody: ...")
    - Forced hashtags or excessive emojis
    - Generic statements anyone could make

    **Output exactly one tweet that feels organic, surprising, and immediately retweetable.**""")
    ]
    response = generator_llm.invoke(message).content
    return {'tweet':response}

def evaluate_tweet(state:postState) -> postState:

    message = [
    SystemMessage(content="""You are a veteran social media strategist and humor critic who has analyzed thousands of viral tweets. You evaluate content with brutal honesty, combining data-driven intuition with creative expertise. You don't just judge—you diagnose why content works or fails, focusing on psychological triggers, platform dynamics, and audience behavior."""),

    HumanMessage(content=f"""Evaluate this tweet for viral potential:

"{state['tweet']}"

## EVALUATION FRAMEWORK

**1. ORIGINALITY SCORE (1-10)**
- Is this perspective unique or just a repackaged common take?
- Does it offer novel insight, unexpected connections, or fresh framing?
- Does it avoid predictable patterns and overused formats?

**2. HUMOR EFFECTIVENESS (1-10)**
- **Relatability**: Does it tap into universal but specific experiences?
- **Surprise**: Does the humor come from unexpected juxtaposition or timing?
- **Economy**: Is every word serving the humor? No wasted setup.
- **Tone**: Does it match conversational, natural human expression?

**3. VIRAL MECHANICS (1-10)**
- **Share Motivation**: Would someone share to:
  - Express identity ("this is so me")
  - Connect with others ("tagging my friend...")
  - Signal insight ("this is brilliant")
  - React to culture ("exactly how it feels")
- **Platform Optimization**: Does it work within Twitter/X's specific dynamics?
- **Engagement Hooks**: Does it invite likes, retweets, or replies naturally?

**4. STRUCTURAL INTEGRITY**
- **Length**: Under 265 chars (allows for RT comments)
- **Flow**: Natural reading rhythm, not forced
- **Ending Impact**: Final line delivers the strongest element
- **Readability**: Scannable at a glance

## AUTO-REJECT CRITERIA (Any one fails → "rejected")
❌ Question-answer format or rhetorical question setups
❌ Exceeds 280 characters
❌ Traditional joke structure (setup → punchline)
❌ Contains "Nobody:" or overused meme templates
❌ Ends with weak, explanatory, or deflating lines
❌ Sounds like AI-generated "try-hard" humor
❌ Uses excessive hashtags/emojis that feel inorganic

## APPROVAL THRESHOLDS
- **Approved**: Scores 7+ across all categories, passes auto-reject
- **Needs Improvement**: Fails one scoring category (5-6) but has strong potential
- **Rejected**: Fails auto-reject or scores below 5 in any category

## RESPONSE FORMAT
**Verdict:** [approved/needs_improvement/rejected]

**Scores:**
- Originality: X/10
- Humor: X/10  
- Virality: X/10
- Structure: [pass/fail]

**Diagnosis:**
[One paragraph analyzing the tweet's strongest viral mechanism and most critical weakness]

**Prescription:** (Only if "needs_improvement")
[Specific, actionable rewrite suggestion targeting the weakest element]

**Viral Trigger Identification:**
[Which psychological share motivation this tweet primarily activates: identity expression, social connection, insight signaling, or cultural reaction]

---
Evaluate ruthlessly. Good tweets are rare.""")
]
    
    response = structred_model.invoke(message)
    return {'evaulation':response.evaluation , 'feedback':response.feedback}
     

def optimize_tweet(state:postState) -> postState:
    message = [
    SystemMessage(content="""You are a top-tier tweet doctor and viral content surgeon. You specialize in transforming good ideas into shareable, culturally-relevant content by applying platform psychology, comedic timing, and audience behavior insights. Your rewrites preserve the core idea while elevating execution for maximum engagement."""),

    HumanMessage(content=f"""## TWEET SURGERY BRIEF

**ORIGINAL TWEET:**
"{state['tweet']}"

**TOPIC CONTEXT:**
{state['topic']}

**DIAGNOSIS (Feedback):**
{state['feedback']}

---

## OPERATING PRINCIPLES

1. **PRESERVE THE CORE INSIGHT**: Keep the original's fundamental observation or humor premise
2. **OPTIMIZE FOR SCANNING**: First line must hook, middle delivers, last line resonates
3. **AMPLIFY EMOTIONAL TRIGGERS**: Strengthen the psychological share motive (identity, connection, reaction, insight)
4. **APPLY COMEDIC PRECISION**: 
   - Trim unnecessary words
   - Improve timing and rhythm
   - Sharpen specific details
   - Strengthen the ending
5. **RESPECT PLATFORM DYNAMICS**: 
   - 260 character max (for RT space)
   - Natural conversation flow
   - No forced hashtags/emojis

---

## REWRITE STRATEGY SELECTOR

Based on the feedback, apply **ONE PRIMARY TECHNIQUE**:

**A. DENSIFICATION** (If feedback mentions "wordy" or "could be tighter")
- Remove filler words and phrases
- Convert clauses to single impactful words
- Use stronger, more specific verbs

**B. SPECIFICITY BOOST** (If feedback mentions "vague" or "generic")
- Replace general statements with hyper-specific details
- Add concrete examples that create visual imagery
- Use numbers, names, or specific scenarios

**C. ENDING SURGERY** (If feedback mentions "weak ending" or "fizzles")
- Move strongest element to final position
- Cut explanatory or summary lines
- End on the punchline or insight, not setup

**D. TONE CALIBRATION** (If feedback mentions "sounds AI" or "forced")
- Inject natural human speech patterns
- Add conversational fragments ("like," "you know," "I mean")
- Use informal contractions and realistic phrasing

**E. STRUCTURE REBUILD** (If feedback mentions "bad flow" or "awkward")
- Reorder elements for better comedic timing
- Break into natural thought units
- Create better rhythm and cadence

---

## CONSTRAINTS
- ❌ NO question-answer formats
- ❌ NO "Nobody:" or overused meme templates  
- ❌ NO setup-punchline joke structure
- ❌ NO rhetorical questions as hooks
- ❌ NO brand voice or corporate tone
- ✅ MUST sound like one human tweeting to another
- ✅ MUST end with strength (punchline or resonant insight)
- ✅ MUST be 260 characters or less

---

## OUTPUT FORMAT

**Improved Tweet:**
[Your rewritten version]

**Technique Applied:**
[Which primary technique you used: Densification / Specificity Boost / Ending Surgery / Tone Calibration / Structure Rebuild]

**Key Change:**
[One sentence explaining the most impactful improvement]

**Character Count:**
[Number] / 260
""")
]

    response = optimizer_llm.invoke(message).content
    state['iteration']=state.setdefault('iteration',0)
    state['max_iteration']=state.setdefault('max_iteration',5)
    iteration = state['iteration'] + 1

    return{'tweet':response ,'iteration':iteration}

def check_condition(state:postState):
    if state['evaulation']=='approved' and state['max_iteration'] >= state['iteration']:
        return END
    else :
        return 'optimize_tweet'




graph = StateGraph(postState)


graph.add_node('generate_tweet',generate_tweet)
graph.add_node('evaulate_tweet',evaluate_tweet)
graph.add_node('optimize_tweet',optimize_tweet)

graph.add_edge(START,'generate_tweet')
graph.add_edge('generate_tweet','evaulate_tweet')
graph.add_conditional_edges('evaulate_tweet',check_condition)

graph.add_edge('optimize_tweet','evaulate_tweet')

workflow = graph.compile()

initial_state = {
    'topic':"Inter-Services Intelligence responsible for many terrorist attack in India ",
    "iteration":1,
    "max_iteration":5
}

result = workflow.invoke(initial_state)
print(f"Result : ", result)

