# share_app.py - Streaming chatbot for stock market with model and word control

import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Load environment variables ---
load_dotenv('./../.env')

# --- Available local models ---
AVAILABLE_MODELS = [
    "llama3.2:3b",
    "llama3.2:1b",
    "deepseek-coder:latest",
    "qwen2-math:1.5b"
]

# --- News URLs ---
urls = [
    'https://economictimes.indiatimes.com/markets/stocks/news',
    'https://www.livemint.com/latest-news',
    'https://www.livemint.com/latest-news/page-2',
    'https://www.livemint.com/latest-news/page-3',
    'https://www.moneycontrol.com/',
]

# --- Load & Format Data ---
loader = WebBaseLoader(web_paths=urls)
docs = []

def load_data():
    for doc in loader.load():
        docs.append(doc)

def format_docs(docs):
    raw = "\n\n".join([x.page_content for x in docs])
    text = re.sub(r'\n\n+', '\n\n', raw)
    text = re.sub(r'\t+', '\t', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Streamlit UI Setup ---
st.set_page_config(page_title="📈 Share Market Chatbot", layout="wide")
st.image("HKA_LOGO.png", width=700)
st.title("📈 Share Market Chatbot: Global Cues Analysis")

# --- Sidebar: model & word limit selection ---
st.sidebar.header("Settings")
selected_model = st.sidebar.selectbox("Choose Model", AVAILABLE_MODELS, index=0)
word_limit = st.sidebar.slider("Select Response Word Limit", min_value=50, max_value=1000, step=50, value=300)

# --- Sidebar: New Conversation Button ---
if st.sidebar.button("🔄 Start New Conversation"):
    st.session_state.messages = []
    st.rerun()  # updated from st.experimental_rerun()

# --- Chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Initialize LLM with selected model and streaming ---
llm = ChatOllama(model=selected_model, streaming=True)

# --- Prompt setup with word limit ---
system = SystemMessagePromptTemplate.from_template(
    "You are an expert AI stock analyst. Use the context to answer financial questions clearly and accurately."
)
human = HumanMessagePromptTemplate.from_template(
    "Context:\n{context}\n\nQuestion: {question}\n\nPlease answer in no more than {words} words."
)
chat_prompt = ChatPromptTemplate.from_messages([system, human])
chain = chat_prompt | llm | StrOutputParser()

# --- Chat input ---
user_query = st.chat_input("Ask a market-related question...")

# --- Render history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Handle new query ---
if user_query:
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    if not docs:
        with st.spinner("📰 Fetching latest news..."):
            load_data()
    context = format_docs(docs)

    with st.chat_message("assistant"):
        try:
            stream = chain.stream({
                "context": context,
                "question": user_query,
                "words": word_limit
            })
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"❌ Error: {e}")
