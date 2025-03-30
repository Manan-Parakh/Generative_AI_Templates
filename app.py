import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import os

import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGSMITH_PROJECT"]=os.getenv("LANGSMITH_PROJECT")

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,api_key,llm,temperature,max_tokens):
    
    if (llm == 'gemma3:1b'):
        llm = Ollama(model=llm)
    else:
        llm = ChatOpenAI(model=llm, openai_api_key=api_key)
    output_parser=StrOutputParser()
    chain=prompt|llm|output_parser
    answer=chain.invoke({'question':question})
    return answer

## Title of the app
st.title("Enhanced Q&A Chatbot With OpenAI/Ollama")
## Select the model provider first
model_provider = st.sidebar.selectbox("Select Model Provider", ["OpenAI", "Ollama"])

if model_provider == "OpenAI":
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    engine = st.sidebar.selectbox("Select OpenAI model", ["gpt-4", "gpt-4-turbo", "gpt-4"])
else:
    api_key = None  # No API key needed for Ollama
    engine = st.sidebar.selectbox("Select Ollama model", ["gemma3:1b"])

## Adjust response parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    if model_provider == "OpenAI" and not api_key:
        st.warning("Please enter the OpenAI API Key in the sidebar")
    else:
        response = generate_response(user_input, api_key, engine, temperature, max_tokens)
        st.write(response)
else:
    st.write("Please provide the user input")
