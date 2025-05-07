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

# Load .env
load_dotenv()

# Streamlit Setup
st.title('Conversational RAG with PDF Upload and Message History')
st.write('Upload PDFs and chat with their content')

# Input API key
api_key = st.text_input("Enter your Groq API Key:", type='password')

if api_key:
    # Initialize LLM and Embeddings
    llm = ChatGroq(groq_api_key=api_key, model_name="gemma2-9b-it")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    session_id = st.text_input("Session ID:", value="default_session")

    # Chat history store in session_state
    if 'store' not in st.session_state:
        st.session_state.store = {}

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
            "You are a helpful assistant for question-answering tasks.\n"
            "Use the following retrieved context to answer the question.\n"
            "If unsure, say 'I don't know'.\n"
            "Answer in under 3 sentences.\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ('system', system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human', '{input}')
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        # Input
        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {'input': user_input},
                config={
                    'configurable': {'session_id': session_id}
                }
            )
            st.write("Assistant:", response['answer'])
            st.markdown("### Chat History")
            for msg in session_history.messages:
                st.markdown(f"- **{msg.type.title()}**: {msg.content}")

else:
    st.warning('Please enter your GROQ API Key to continue.')
