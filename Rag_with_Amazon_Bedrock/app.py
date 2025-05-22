import streamlit as st
import boto3
from langchain.embeddings import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os

# Caching LLM and embeddings
@st.cache_resource
def get_llm_client_and_embeddings():
    bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
    bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0", client=bedrock)
    llm = ChatBedrock(
        model_id="amazon.nova-lite-v1:0",
        client=bedrock
    )
    return llm, bedrock_embeddings

# Prompt template
prompt_template = '''
Use the following piece of context to answer the given question.
If you don't know, simply say so. Answer based on the context only.
Context: {context}
Question: {question}
Assistant:
'''
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# QA retrieval chain
def get_response(llm, vectorstore, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    answer = qa({"query": query})
    return answer["result"]

# PDF handling and vectorstore creation
def process_documents(uploaded_files, embeddings):
    documents = []
    for uploaded_file in uploaded_files:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader("temp.pdf")
        documents.extend(loader.load())

    st.info(f"Loaded {len(documents)} total pages from {len(uploaded_files)} file(s).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_documents(documents)
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)

    return vectordb

# MAIN APP
def main():
    st.set_page_config(page_title="RAG Q&A with Langchain and Bedrock", page_icon="📄")
    st.title("RAG Q&A with Langchain and Bedrock")

    llm, embeddings = get_llm_client_and_embeddings()

    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        if "vectordb" not in st.session_state:
            st.session_state.vectordb = process_documents(uploaded_files, embeddings)

    if "vectordb" in st.session_state:
        user_question = st.text_input("Ask a question based on the uploaded PDFs:")

        if user_question:
            with st.spinner("Thinking..."):
                response = get_response(llm, st.session_state.vectordb, user_question)
                st.success(response)
if __name__ == "__main__":
    main()

