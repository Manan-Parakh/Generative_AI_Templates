# basic imports
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler
# the llm and chains
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
# for prompts
from langchain.prompts import PromptTemplate
# for tools and agent creation
from langchain_community.utilities import WikipediaAPIWrapper 
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langsmith import Client
import os

client = Client(api_key = os.getenv("LANGSMITH_API_KEY_TEXT_TO_MATH"))
# Setup the streamlit app
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant", page_icon="🧮")
st.title('Text to Math Problem Solver')

# setup the llm
groq_api_key = st.sidebar.text_input('Groq API Key', type='password')
if not groq_api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()
llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

# tool creation
##wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
##wikipedia_tool = Tool(name = "Wikipedia", func=wikipedia_wrapper.run, description="Use this tool to search for the required formulas and information about various topics.")
### Wikipedia wrapper is not fetching the formula directly leading to multiple redundant calls
### Let's use RESTAPI to fetch the whole page and get the formula enclosed in <math> tags
import requests
from bs4 import BeautifulSoup
def fetch_wikipedia_formulas(topic: str) ->str:
    """
    Fetch Latex math formulas for the given topic using REST API
    """
    url = f"https://en.wikipedia.org/api/rest_v1/page/html/{topic.replace(' ', '_')}"
    respone = requests.get(url)

    if respone.status_code != 200:
        return f'Failed to fetch the wikipedia page for "{topic}"'
    
    soup = BeautifulSoup(respone.text, "html.parser")
    formulas = [math.text.strip() for math in soup.find_all("math")]

    if not formulas:
        return f"No formulas found for '{topic}'"
    
    return "\n\n".join(formulas[:10]) # Return the first 10 formulas

# wrap in a langchain tool
wikipedia_formula_tool  =Tool(
    name = 'wikipedia_formula_tool',
    description= """use this tool to get the formulas for any given topic. Input should be a string with just the topic name.
    If failed, try again by rephrasing the topic name to a more general name.""",
    func = fetch_wikipedia_formulas
)

# initialize the math tool - for solving the problem
math_chain = LLMMathChain.from_llm(llm=llm)

calculator = Tool(name = "Calculator",
                  func = math_chain.run,
                  description="Use this tool for answering math related problems. Only input Mathematical Expressions.")

# prompt creation
prompt = """"
You are an agent which is supposed to solve the mathematical questions provided by the user.
Logically arrive at the solution and provide a detailed solution explaining the thought process pointwise.
Question:{question}
Answer:
"""
prompt_template = PromptTemplate(template = prompt,
                                 input_variables=['question'])
# Create a chain combining all the tools
chain = LLMChain(llm = llm, prompt = prompt_template)
# To add reasoning
reasoning_tool = Tool(name = 'Reasoning Tool',
                      func = chain.run,
                      description='A tool for answering logic based and reasoning questions.')

# Initialize the agent
assistant_agent = initialize_agent(
    tools = [wikipedia_formula_tool, calculator, reasoning_tool],
    llm=llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose =False,
    handle_parsing_errors = True
)

if "messages" not in st.session_state:
    st.session_state['messages'] = [{'role':'assistant', 'content':'Hi, I am a Math Chatbot. How can I help you today?'}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# Start the interaction
question=st.text_area("Enter your question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")
if st.button('Solve the Question'):
    if question:
        with st.spinner('Solving the questions...'):
            # Append the input to the session history
            st.session_state.messages.append({'role':'user','content':question})
            with st.chat_message('assistant'):
                # Fetch the output of the question
                response = assistant_agent.invoke(question, callbacks= [StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)])
                # print the response and als add it to the session history
                st.session_state.messages.append({'role':'assistant','content':response})
                st.write(response)