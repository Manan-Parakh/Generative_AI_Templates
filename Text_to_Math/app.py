# basic imports
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler
from langsmith.tracing import LangChainTracer  # ✅ LangSmith Tracing Import
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper 
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
import os
import requests
from bs4 import BeautifulSoup

# Optional: for local development
load_dotenv()

# ✅ LangSmith environment setup
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGSMITH_TRACING"]
os.environ["LANGCHAIN_ENDPOINT"] = st.secrets["LANGSMITH_ENDPOINT"]
os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGSMITH_PROJECT"]

# Streamlit UI setup
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant", page_icon="🧮")
st.title('Text to Math Problem Solver')

# Get Groq API key
groq_api_key = st.sidebar.text_input('Groq API Key', type='password')
if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

# Set up LLM
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Wikipedia Formula Fetcher
def fetch_wikipedia_formulas(topic: str) -> str:
    url = f"https://en.wikipedia.org/api/rest_v1/page/html/{topic.replace(' ', '_')}"
    respone = requests.get(url)
    if respone.status_code != 200:
        return f'Failed to fetch the Wikipedia page for "{topic}"'
    soup = BeautifulSoup(respone.text, "html.parser")
    formulas = [math.text.strip() for math in soup.find_all("math")]
    if not formulas:
        return f"No formulas found for '{topic}'"
    return "\n\n".join(formulas[:10])

wikipedia_formula_tool = Tool(
    name='wikipedia_formula_tool',
    func=fetch_wikipedia_formulas,
    description="""use this tool to get the formulas for any given topic. 
    Input should be a string with just the topic name. 
    If failed, try again by rephrasing the topic name to a more general name."""
)

# Math tool
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Use this tool for answering math-related problems. Only input mathematical expressions."
)

# Prompt & Reasoning Chain
prompt = """
You are an agent which is supposed to solve the mathematical questions provided by the user.
Logically arrive at the solution and provide a detailed solution explaining the thought process pointwise.
Question: {question}
Answer:
"""
prompt_template = PromptTemplate(template=prompt, input_variables=['question'])
chain = LLMChain(llm=llm, prompt=prompt_template)
reasoning_tool = Tool(
    name='Reasoning Tool',
    func=chain.run,
    description='A tool for answering logic-based and reasoning questions.'
)

# Initialize Agent
assistant_agent = initialize_agent(
    tools=[wikipedia_formula_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Chat history
if "messages" not in st.session_state:
    st.session_state['messages'] = [{'role': 'assistant', 'content': 'Hi, I am a Math Chatbot. How can I help you today?'}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# User Input
question = st.text_area("Enter your question:", "I have 5 bananas and 7 grapes...")

if st.button('Solve the Question'):
    if question:
        with st.spinner('Solving the question...'):
            st.session_state.messages.append({'role': 'user', 'content': question})
            with st.chat_message('assistant'):
                # ✅ Use LangSmith tracer and StreamlitCallbackHandler
                tracer = LangChainTracer()
                response = assistant_agent.invoke(
                    question,
                    callbacks=[
                        StreamlitCallbackHandler(st.container(), expand_new_thoughts=True),
                        tracer
                    ]
                )
                st.session_state.messages.append({'role': 'assistant', 'content': response})
                st.write(response)