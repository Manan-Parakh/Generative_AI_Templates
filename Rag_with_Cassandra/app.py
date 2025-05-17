import streamlit as st
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.vectorstores.cassandra import Cassandra
import cassio
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.schema import AIMessage, HumanMessage
from langchain.memory import ChatMessageHistory

# Setup the page
st.set_page_config(page_title="Conversational RAG with PDF Upload & Cassandra Vector Store", page_icon="🤖")
st.title('🤖 Conversational RAG with PDF Upload & Cassandra Vector Store')
st.sidebar.header('Please Enter the Keys')

# Sidebar inputs
openai_api_key = st.sidebar.text_input('Enter your OpenAI API Key', type='password')
astradb_application_token = st.sidebar.text_input('Enter your ASTRA_DB_APPLICATION_TOKEN', type='password')
astradb_id = st.sidebar.text_input('Enter your ASTRA_DB_ID', type='password')
table_name = st.sidebar.text_input('Enter a Table Name to store the data')
# Initialize LLM, embeddings, and Cassandra vector store
@st.cache_resource
def llm_embedding_and_database(openai_api_key, astradb_application_token, astradb_id, table_name):
    llm = OpenAI(api_key=openai_api_key)
    embedding = OpenAIEmbeddings(api_key=openai_api_key)
    cassio.init(token=astradb_application_token, database_id=astradb_id)
    astra_vector_store = Cassandra(embedding=embedding, table_name=table_name)
    return llm, embedding, astra_vector_store

initialize_clicked = st.sidebar.button('Initialize the Setup')

if initialize_clicked:
    missing_fields = []
    if not openai_api_key:
        missing_fields.append("OpenAI API Key")
    if not astradb_application_token:
        missing_fields.append("ASTRA_DB_APPLICATION_TOKEN")
    if not astradb_id:
        missing_fields.append("ASTRA_DB_ID")
    if not table_name:
        missing_fields.append("Table Name")
    if missing_fields:
        st.warning(f"Please provide all the required values! Missing: {', '.join(missing_fields)}")
        st.stop()
    else:
        llm, embedding, astra_vector_store = llm_embedding_and_database(
            openai_api_key, astradb_application_token, astradb_id, table_name
        )
        st.success("Setup Initialized Successfully! Now upload PDFs and click 'Process Documents'.")
        
with st.expander("Upload Documents to be Used", expanded=False):
    # Upload PDFs
    uploaded_files = st.file_uploader("Upload the PDF(s)", accept_multiple_files=True, type='pdf')

    # Store loaded documents temporarily in session state to avoid reloading on every rerun
    if 'loaded_documents' not in st.session_state:
        st.session_state.loaded_documents = []

    if uploaded_files:
        # Save uploaded files and load documents
        documents = []
        for uploaded_file in uploaded_files:
            with open('./temp.pdf', 'wb') as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader('./temp.pdf')
            documents.extend(loader.load())
        st.session_state.loaded_documents = documents
        st.info(f"Uploaded {len(uploaded_files)} file(s) with {len(documents)} total pages loaded.")

    # Process documents button
    if st.button("Process Documents"):
        if 'astra_vector_store' not in st.session_state:
            st.warning("Please initialize the setup first!")
            st.stop()
        if not st.session_state.loaded_documents:
            st.warning("Please upload at least one document before processing.")
            st.stop()

        with st.spinner("Processing documents and storing in vector DB..."):
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(st.session_state.loaded_documents)
            st.session_state.astra_vector_store.add_documents(splits)
            st.success(f"Processed and stored {len(splits)} document chunks in vector DB.")

# Ensure we have initialized resources before continuing
if 'llm' in locals() and 'astra_vector_store' in locals():
    st.session_state.llm = llm
    st.session_state.astra_vector_store = astra_vector_store

if 'llm' in st.session_state and 'astra_vector_store' in st.session_state:
    # Retriever setup
    retriever = st.session_state.astra_vector_store.as_retriever()

    # History-aware retriever
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ('system', """Given a chat history and latest query, which might refer to something in the chat history, 
                    create a standalone question. Do not answer the question. Just reformulate."""),
        MessagesPlaceholder('chat_history'),
        ('human', '{input}')
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm=st.session_state.llm,
        prompt=contextualize_q_prompt,
        retriever=retriever
    )

    # QA chain setup
    qa_prompt = ChatPromptTemplate.from_messages([
        ('system', """You are a helpful RAG assistant. Refer to the context while answering the user's question.
                    Don't answer outside the context unless explicitly asked. {context}"""),
        MessagesPlaceholder('chat_history'),
        ('user', '{input}')
    ])
    question_answer_chain = create_stuff_documents_chain(st.session_state.llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Initialize session message state for chat
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{'role': 'assistant', 'content': "Hey! Let's discuss the documents!"}]

    # Convert session state messages to LangChain chat history
    def session_state_to_chat_history():
        chat_history = ChatMessageHistory()
        for msg in st.session_state['messages']:
            if msg['role'] == 'user':
                chat_history.add_message(HumanMessage(content=msg['content']))
            else:
                chat_history.add_message(AIMessage(content=msg['content']))
        return chat_history

    # Create history-aware RAG chain with memory
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: session_state_to_chat_history(),
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
    )

    # Display chat messages
    for msg in st.session_state['messages']:
        st.chat_message(msg['role']).write(msg['content'])

    # Chat input
    user_query = st.chat_input("Ask a question about the documents:")

    if user_query:
        st.session_state['messages'].append({'role': 'user', 'content': user_query})
        st.chat_message("user").write(user_query)

        with st.spinner("Reading the Documents..."):
            response = conversational_rag_chain.invoke({"input": user_query}, config={"configurable": {"session_id": "any"}})
            answer = response['answer']

        st.session_state['messages'].append({'role': 'assistant', 'content': answer})
        st.chat_message("assistant").write(answer)
else:
    st.info("Please initialize the setup first from the sidebar.")
