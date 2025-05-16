# 🧠 Generative AI Templates

A collection of plug-and-play generative AI mini-projects, including a SQL chatbot, webpage summarizer, and AI-powered search engine — built using LangChain, Streamlit, and other LLM tools.

## 🚀 Projects Included

### 1. 💬 Chat_SQL - [Streamlit app](https://langchain-chatsql.streamlit.app/)
> A natural language interface to interact with SQL databases using LangChain agents.

- Input: User prompt (e.g., "Get me all rows where revenue > 1 million")
- Output: Corresponding SQL query + result fetched from the database
- Technologies: LangChain, SQLAlchemy, Streamlit

### 2. 📰 Text_Summarization - [Streamlit app](https://langchain-text-summarizer.streamlit.app/)
> Summarize lengthy web pages into concise insights.

- Input: URL of a webpage
- Output: Bullet-point or paragraph summary
- Technologies: LangChain, WebPageLoader, LLMs

### 3. 🔍 Search_Engine
> Query a multi-tool generative agent that can access the web and knowledge sources.

- Tools used:
  - Wikipedia
  - Arxiv
  - DuckDuckGoSearchRun
- Input: Natural language query
- Output: Collated answer after web + document retrieval
- Technologies: LangChain Agents, Multiple Toolkits

---

## 🧰 Tech Stack

- Python
- LangChain
- Streamlit
- OpenAI / LLM APIs
- FAISS
- BeautifulSoup
- SQLAlchemy
- Wikipedia, Arxiv, DuckDuckGo APIs

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Manan-Parakh/Generative_AI_Templates.git
cd Generative_AI_Templates

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

