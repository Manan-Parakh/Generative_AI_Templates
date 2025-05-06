import streamlit as st
import os
import time
import tempfile
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("Missing GROQ_API_KEY in .env file.")
    st.stop()

# Set up the LLM from Groq
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.

    <context>
    {context}
    <context>

    Question: {input}
    """
)

# Title
st.title("📄 RAG Document Q&A with Groq + LLaMA3 (Open Source)")

# File uploader
uploaded_files = st.file_uploader("Upload one or more PDF documents", type=["pdf"], accept_multiple_files=True)

# Create vector embeddings from uploaded PDFs
def create_vector_embedding(uploaded_files):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
        return

    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        all_docs.extend(docs)

    if not all_docs:
        st.warning("No readable content found in uploaded PDFs.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    st.session_state.vectors = FAISS.from_documents(split_docs, embeddings)
    st.success("✅ Vector database created successfully from uploaded PDFs.")

# Embed button
if st.button("📚 Create Vector Database"):
    create_vector_embedding(uploaded_files)

# User input
user_prompt = st.text_input("🔎 Ask a question about your PDFs:")

# Run retrieval + generation
if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("Please upload and embed documents first.")
        st.stop()

    retriever = st.session_state.vectors.as_retriever()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    with st.spinner("Generating response..."):
        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        end = time.process_time()

    st.subheader("🧠 Answer:")
    st.write(response.get('answer', 'No answer returned by the model.'))
    st.caption(f"⏱️ Took {end - start:.2f} seconds")

    with st.expander("📄 Retrieved Context"):
        context = response.get('context', [])
        if not context:
            st.write("No relevant context retrieved.")
        else:
            for i, doc in enumerate(context):
                st.markdown(f"**Document {i+1}:**")
                st.write(doc.page_content)
                st.markdown("---")
