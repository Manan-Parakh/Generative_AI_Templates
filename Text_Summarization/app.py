import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Strealit Setup
st.set_page_config(page_title="LangChain: Summarize Text From a Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From a Website")
st.subheader('Summarize URL')

# Get the Groq Api Key and the URL to be summarized
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

## Gemma Model USsing Groq API
if groq_api_key:
    llm =ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)

prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button('Summarize the content'):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide all the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url!")
    else:
        try:
            with st.spinner('Summarizing'):
                loader = UnstructuredURLLoader(urls = [generic_url],
                                                   ssl_verify = True,
                                                   headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs = loader.load()
                # Split the text into chunks
                splits = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap = 200).split_documents(docs)
                # Chain for summarizing
                summarizing_chain = load_summarize_chain(llm, chain_type='refine')
                output_summary=summarizing_chain.run(splits)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")
