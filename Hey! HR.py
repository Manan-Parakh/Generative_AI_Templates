# Basic Library Imports
import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

from chromadb.config import Settings as ChromaSettings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import json

# Load .env
load_dotenv()

# Streamlit Setup
st.title('Conversational RAG with PDF Upload and Message History')
st.write('Upload PDFs and chat with their content')

# Input API key
api_key = st.text_input("Enter your Groq API Key:", type='password', value = "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")

if api_key:
    # Initialize LLM and Embeddings
    llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    session_id = st.text_input("Session ID:", value="default_session")

# Initialize store if not exists
    if 'store' not in st.session_state:
        st.session_state.store = {}

    # Function to save chat history to file
    def save_history_to_file(session: str, messages: List[BaseMessage]):
        history_dir = "./chat_histories"
        os.makedirs(history_dir, exist_ok=True)
        
        history_file = os.path.join(history_dir, f"{session}.json")
        messages_dict = [{"type": msg.type, "content": msg.content} for msg in messages]
        
        with open(history_file, "w") as f:
            json.dump(messages_dict, f, indent=4, ensure_ascii=False, separators=(',\n', ': '))
    
    # Function to load chat history from file
    def load_history_from_file(session: str) -> List[BaseMessage]:
        history_file = os.path.join("./chat_histories", f"{session}.json")
        if not os.path.exists(history_file):
            return []
            
        with open(history_file, "r") as f:
            messages_dict = json.load(f)
            
        return [HumanMessage(content=msg["content"]) if msg["type"] == "human" 
                else AIMessage(content=msg["content"]) for msg in messages_dict]
    
    # get_session_history function
    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            chat_history = ChatMessageHistory()
            loaded_messages = load_history_from_file(session)
            chat_history.messages = loaded_messages
            st.session_state.store[session] = chat_history
        return st.session_state.store[session]
    
    # Display chat history when session ID is entered, even before asking questions
    if session_id:
        session_history = get_session_history(session_id)
        if session_history.messages:
            st.markdown("### Previous Chat History")
            for msg in session_history.messages:
                st.markdown(f"- **{msg.type.title()}**: {msg.content}")
        else:
            st.info("No previous chat history found for this session ID.")
    
    # Add a reset button for chat history
    if st.button("Reset Chat History"):
        if session_id:
            if session_id in st.session_state.store:
                del st.session_state.store[session_id]
            # Also delete the file if it exists
            history_file = f"chat_histories/{session_id}.json"
            if os.path.exists(history_file):
                os.remove(history_file)
            st.success(f"Chat history for session '{session_id}' has been reset.")
            st.rerun()  # This will refresh the page and update the displayed chat history
        else:
            st.warning("Please enter a session ID first.")
    
    uploaded_files = st.file_uploader('Upload PDF Files', type='pdf', accept_multiple_files=True)
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            with open('./temp.pdf', 'wb') as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader('./temp.pdf')
            documents.extend(loader.load())

        # Text splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)

        # Chroma settings
        chroma_settings = ChromaSettings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
            # Do not include tenant or database if using ChromaDB < 0.4.0
        )

        # Vectorstore creation
        vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
        )

        retriever = vectorstore.as_retriever()

        # History-aware retriever setup
        contextualize_q_system_prompt = """Given a chat history and latest user question,
        which might refer to something in the chat history,
        formulate a standalone question which can be understood
        without the chat history. Do not answer the question,
        just reformulate the question if needed; otherwise,
        return it as-is."""

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ('system', contextualize_q_system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}')
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # QA Prompt
        system_prompt = (
            "You are a helpful assistant whose primary task is to introduce me to the HR talking to you.\n"
            "I will be using this chatbot for company officials to talk about me and decide whether to take me or not\n"
            "Convince them.\n"
            "Answer in under 5 sentences.\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ('system', system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}')
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        # Input
        user_input = st.text_input("Your question:")
        # Save chat history after each interaction
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {'input': user_input},
                config={
                    'configurable': {'session_id': session_id}
                }
            )
            st.write("Assistant:", response['answer'])
            save_history_to_file(session_id, session_history.messages)
            st.markdown("### Chat History")
            for msg in session_history.messages:
                st.markdown(f"- **{msg.type.title()}**: {msg.content}")

else:
    st.warning('Please enter your GROQ API Key to continue.')
