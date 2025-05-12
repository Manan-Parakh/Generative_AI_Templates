import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase

from langchain.agents import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

# Setup the page config
st.set_page_config(page_title="Langchain: Chat with SQL DB", page_icon="🦜")
st.title("Langchain: Chat with SQL DB")

# Global Variables
LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"
########################################################################################################################################
################################################### Connecting the Database ############################################################
########################################################################################################################################
# Get the GROQ Api key
api_key = st.sidebar.text_input('Enter your GROQ API Key',type='password')

# Radio option in the sidebar
radio_opt = ['Use SQLITE3 Database: student.db', "Connect to your SQL Database"]
selected_opt = st.sidebar.radio(label = "Choose the DB which you want to use!", options= radio_opt)

if radio_opt.index(selected_opt)==1:
    db_uri=MYSQL
    mysql_host=st.sidebar.text_input("Provide MySQL Host")
    mysql_user=st.sidebar.text_input("MYSQL User")
    mysql_password=st.sidebar.text_input("MYSQL password",type="password")
    mysql_db=st.sidebar.text_input("MySQL database")
else:
    db_uri = LOCALDB

# The warnings
if not db_uri:
    st.info('Please Enter the Database Infomation and URI')
if not api_key:
    st.info('Please Enter the Groq API Key!')

# Call the llm model
if api_key and db_uri:
    llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)


    # Cache the Database
    @st.cache_resource(ttl='2h') # Total time limit
    def configure_db(db_uri,mysql_host=None,mysql_user=None,mysql_password=None,mysql_db=None):
        """
        Configure and return a SQLDatabase connection based on the provided database URI.
        
        Args:
            db_uri (str): Database URI indicating which database to use (LOCALDB or MYSQL)
            mysql_host (str, optional): MySQL host address
            mysql_user (str, optional): MySQL username
            mysql_password (str, optional): MySQL password  
            mysql_db (str, optional): MySQL database name
        
        Returns:
            SQLDatabase: A configured SQLDatabase instance
            
        For local SQLite database:
        - Gets absolute path to student.db in same directory as this file
        - Creates read-only connection to SQLite database
        - Returns SQLDatabase wrapper around SQLite engine
        """
        if db_uri == LOCALDB:
            dbfilepath = (Path(__file__).parent/"student.db").absolute()  ## Get absolute path to DB
            print(dbfilepath)
            creator = lambda: sqlite3.connect(f'file:{dbfilepath}?mode=ro', uri=True)  # Read-only connection
            return SQLDatabase(create_engine("sqlite:///", creator=creator))
            # For SQLite:
            # - creator is a lambda function that creates a read-only SQLite connection
            # - create_engine creates a SQLAlchemy engine with the creator function
            # - This ensures we have a read-only connection to prevent accidental writes
            
            # We use a lambda function instead of directly writing sqlite3.connect because:
            # 1. SQLAlchemy's create_engine expects a callable (function) for its creator parameter
            # 2. The lambda allows lazy evaluation - the connection is only created when needed
            # 3. This helps with connection pooling and resource management
            # 4. Each time the engine needs a new connection, it calls this lambda

            # If we wrote sqlite3.connect directly, it would execute immediately and only once,
            # rather than creating fresh connections as needed by the engine

            # For MySQL:
            # - create_engine creates a SQLAlchemy engine directly from the connection URI
            # - The engine manages the connection pool and provides database connectivity
            # - No creator function needed since MySQL permissions handle access control
        elif db_uri == MYSQL:
            if not (mysql_host and mysql_user and mysql_password and mysql_db):
                st.error("Please provide all MySQL connection details.")
                st.stop()
            # Connecting to mysql database
            return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))
    # Get the Database
    if db_uri == MYSQL:
        db = configure_db(db_uri, mysql_user=mysql_user, mysql_password=mysql_password, mysql_host=mysql_host, mysql_db=mysql_db)
    else:
        db = configure_db(db_uri)
###############################################################################################################
################################################### Toolkit ###################################################
###############################################################################################################
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent = create_sql_agent(llm = llm, 
                            verbose=True, toolkit=toolkit,
                            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors= True)

    ## Session Management
    if "messages" not in st.session_state or st.sidebar.button('Clear Session History!'):
        st.session_state.messages = [
            {"role":"assistant", "content": "Hey! How can I help you today?"}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_query = st.chat_input(placeholder="Ask anything about the database!")

    if user_query:
        # Append the message to the session state
        st.session_state.messages.append({'role':'user',"content":user_query})
        with st.chat_message("assistant"):
            streamlit_callback=StreamlitCallbackHandler(st.container())
            # The StreamlitCallbackHandler displays the agent's thought process in real-time
            # It shows each step the agent takes while processing the query:
            # - The agent's thoughts and reasoning
            # - Tools/APIs being called
            # - Intermediate results
            # - Any errors or retries
            
            # st.container() creates a new container element in the Streamlit app
            # This container acts as a dedicated space where the callback will display
            # all the intermediate steps and thought process
            # Using a container helps organize the output and keeps it separate from
            # other elements in the UI
            response=agent.run(user_query,callbacks=[streamlit_callback])
            st.session_state.messages.append({"role":"assistant","content":response})
            st.write(response)



    

    