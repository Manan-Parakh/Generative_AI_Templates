import streamlit as st
from langchain_groq import ChatGroq 
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os

# The Tools
## Arxiv and Wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper= wiki_wrapper)
## Tool for searching the web
search = DuckDuckGoSearchRun(name = 'search')

# For the Groq Api Key
st.sidebar.title('Settings')
api_key = st.sidebar.text_input("Enter your Groq API Key", type = "password")

if not api_key:
    st.warning("Please enter your Groq API key to use the assistant.")
    st.stop()

# Streamlit App
st.title("🔎 LangChain - Chat with search")
"""
The Streamlit Callback Handler displays the agent's thought process and actions in real-time 
in the Streamlit interface. It shows each step the agent takes, including:
- The agent's thoughts and reasoning
- Which tools it decides to use
- The results from those tools
- Its final answer
This creates transparency in how the agent arrives at its responses.
"""

if "messages" not in st.session_state:
    st.session_state['messages'] = [{"role":"assistant", "content":"Hi, I am a chatbot who can search the Web! How can I help you today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# Session State Management
# ----------------------
# st.session_state is Streamlit's way to persist data between reruns of the app
# 'messages' stores the chat history as a list of message dictionaries

# If statement: Initialize Chat History
# -----------------------------------
# Checks if this is the first time running the app
# If 'messages' doesn't exist in session state, creates it with a welcome message
# Format: [{"role": "assistant", "content": "welcome message"}]
# This ensures users always see a greeting when they start the app

# For Loop: Display Chat History
# ----------------------------
# Iterates through every message stored in st.session_state.messages
# Each message is a dictionary with 'role' (who sent it - assistant or user)
# and 'content' (the actual message text)
# st.chat_message() creates chat bubbles styled based on the sender
# .write() displays the message content inside the bubble
# This creates the visual chat interface users see

# Prompt
if (prompt := st.chat_input(placeholder="What is Machine Learning?")) and api_key:
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key = api_key, model_name = "Llama3-8b-8192", streaming=True)
    tools = [wiki,arxiv,search]
    
    search_agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors = True)

    with st.chat_message('assistant'):
        with st.spinner("Thinking..."):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(prompt, callbacks=[st_cb])
            st.write(response)
            st.session_state.messages.append({'role': "assistant", "content": response})


