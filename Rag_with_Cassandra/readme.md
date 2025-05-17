# 🤖 Conversational RAG with PDF Upload & Cassandra Vector Store

This is a Streamlit-based application that allows you to **upload PDF documents** and chat with them using **Retrieval-Augmented Generation (RAG)** powered by **OpenAI** and a **Cassandra-based vector store (Astra DB)**.

---

## 📌 Features

* 📥 Upload multiple PDF documents
* 🔍 Automatically splits and stores them in Astra DB using embeddings
* 💬 Chatbot interface powered by OpenAI's LLM with contextual memory
* 🧠 History-aware RAG pipeline
* 🗂 Document processing via button
* ❌ Clear chat or vector DB (coming soon)
* 🖼️ Clean and interactive UI with Streamlit

---

## ⚙️ Tech Stack

- 🧠 **OpenAI GPT & Embeddings** – For natural language understanding and generating vector embeddings  
- 🔗 **LangChain** – Framework for building Retrieval-Augmented Generation (RAG) pipelines and managing chat history  
- 🗃️ **Astra DB (Cassandra)** – Vector database to store and retrieve document embeddings efficiently  
- 📄 **PyPDFLoader** – Library to extract text from PDF files  
- ✂️ **Text Splitter** – Splits documents into semantically meaningful chunks  
- 💬 **LangChain Memory** – Maintains conversation history for context-aware responses  
- 🎛️ **Streamlit** – User-friendly web interface for document upload, initialization, and chatting  
---

## 🧱 Prerequisites

* Python 3.9 or above
* [OpenAI API Key](https://platform.openai.com/account/api-keys)
* [Astra DB credentials](https://docs.datastax.com/en/astra/docs/)

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/Manan-Parakh/pdf-rag-cassandra.git
cd pdf-rag-cassandra
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the app**

```bash
streamlit run app.py
```

---

## 🔑 Setup

On the **sidebar**:

1. Paste your **OpenAI API Key**
2. Paste your **ASTRA\_DB\_APPLICATION\_TOKEN**
3. Paste your **ASTRA\_DB\_ID**
4. Set a **table name** (used in Astra DB to store vector data)
5. Click **“Initialize the Setup”**

---

## 📝 Usage

1. Upload one or more PDFs using the “Upload Documents” section
2. Click **“Process Documents”**
3. Start chatting with the assistant using the input box
4. Messages are context-aware and use chat history
5. Chat state is retained during the session

---

## 📁 Project Structure

```
pdf-rag-cassandra/
│
├── app.py                 # Main Streamlit application
├── readme.md              # This file
└── .gitignore
```

---

## ✅ To-Do (Future Work)

* [ ] Add button to clear vector DB table data
* [ ] Add persistent storage for chat history
* [ ] Enable document tagging and versioning
* [ ] Multi-user session support
