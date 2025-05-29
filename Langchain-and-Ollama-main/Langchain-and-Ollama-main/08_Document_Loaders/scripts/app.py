import streamlit as st
from dotenv import load_dotenv
import os

# Load .env file for environment variables (like API keys if needed)
load_dotenv('./../.env')

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
)
from langchain_core.output_parsers import StrOutputParser

<<<<<<< HEAD
# ---- Load PDFs ----
PDF_ROOT = "D:/ML/HKA_Chatbot/Langchain-and-Ollama-main/Langchain-and-Ollama-main/08_Document_Loaders/rag-dataset"
pdfs = []
for root, dirs, files in os.walk(PDF_ROOT):
    for file in files:
        if file.endswith(".pdf"):
            pdfs.append(os.path.join(root, file))
=======
# --- SETTINGS ---
BASE_URL = "http://localhost:11434"
MODEL = "llama3.2:3b"
PDF_ROOT = r"HKA_Chatbot\Langchain-and-Ollama-main\Langchain-and-Ollama-main\08_Document_Loaders\rag-dataset"
>>>>>>> 5aa87889ac794e815ab5a0f3c6f87138243f4587

docs = []
for pdf in pdfs:
    try:
        loader = PyMuPDFLoader(pdf)
        temp = loader.load()
        docs.extend(temp)
    except Exception as e:
        st.warning(f"Failed to load {pdf}: {e}")

def format_docs(docs):
    return "\n\n".join([x.page_content for x in docs])

context = format_docs(docs)

# ---- Model Selection ----
BASE_URL = "http://localhost:11434"
AVAILABLE_MODELS = [
    "llama3.2:3b",
    "llama3.2:1b",
    "Sheldon:latest",
    "deepseek-r1:1.5b"
]
selected_model = st.sidebar.selectbox("Select LLM Model:", AVAILABLE_MODELS)
llm = ChatOllama(base_url=BASE_URL, model=selected_model)

# ---- Streamlit UI ----


st.image(r"D:\ML\HKA_Chatbot\Langchain-and-Ollama-main\Langchain-and-Ollama-main\08_Document_Loaders\scripts\HKA_LOGO.png", width=600)
st.title("PDF Chatbot: Select Project & Word Limit")

project = st.sidebar.radio(
    "Choose a project:",
    [
        "Question Answering from PDF",
        "PDF Document Summarization",
        "Report Generation from PDF"
    ]
)
words = st.sidebar.slider(
    "Number of words (approx.)", 
    min_value=20, max_value=2000, value=100, step=10
)

# ---- Project Logic ----

if project == "Question Answering from PDF":
    st.header("Ask a Question about the PDF(s)")
    user_question = st.text_input("Enter your question:")
    if not context.strip():
        st.error("No text found in the loaded PDFs! Please check your PDF folder and try again.")
    elif user_question:
        system = SystemMessagePromptTemplate.from_template(
            "You are a helpful AI assistant who answers user questions based on the provided context. Do not answer in more than {words} words."
        )
        prompt = """Answer the user's question based on the provided context ONLY! If you do not know the answer, just say "I don't know".
        ### Context:
        {context}

        ### Question:
        {question}

        ### Answer:"""
        prompt = HumanMessagePromptTemplate.from_template(prompt)
        messages = [system, prompt]
        template = ChatPromptTemplate(messages)
        qna_chain = template | llm | StrOutputParser()
        if st.button("Get Answer"):
            with st.spinner("Thinking..."):
                response = qna_chain.invoke({'context': context, 'question': user_question, 'words': words})
            st.markdown("**Answer:**")
            st.write(response)

elif project == "PDF Document Summarization":
    st.header("Summarize the PDF(s)")
    if not context.strip():
        st.error("No text found in the loaded PDFs! Please check your PDF folder and try again.")
    else:
        system = SystemMessagePromptTemplate.from_template(
            "You are helpful AI assistant who works as document summarizer. You must not hallucinate or provide any false information."
        )
        prompt = """Summarize the given context in {words} words.
        ### Context:
        {context}

        ### Summary:"""
        prompt = HumanMessagePromptTemplate.from_template(prompt)
        messages = [system, prompt]
        template = ChatPromptTemplate(messages)
        summary_chain = template | llm | StrOutputParser()
        if st.button("Summarize PDF(s)"):
            with st.spinner("Summarizing..."):
                response = summary_chain.invoke({'context': context, 'words': words})
            st.markdown("**Summary:**")
            st.write(response)

elif project == "Report Generation from PDF":
    st.header("Generate a Detailed Report from PDF(s)")
    if not context.strip():
        st.error("No text found in the loaded PDFs! Please check your PDF folder and try again.")
    else:
        system = SystemMessagePromptTemplate.from_template(
            "You are helpful AI assistant who generates detailed reports in Markdown from the provided context. Be accurate and detailed."
        )
        prompt = """Provide a detailed report from the provided context. Write answer in Markdown (do not hallucinate).
        ### Context:
        {context}

        ### Report (max {words} words):"""
        prompt = HumanMessagePromptTemplate.from_template(prompt)
        messages = [system, prompt]
        template = ChatPromptTemplate(messages)
        report_chain = template | llm | StrOutputParser()
        if st.button("Generate Report"):
            with st.spinner("Generating report..."):
                response = report_chain.invoke({
                    'context': context,
                    'words': words,
                })
            st.markdown("**Report:**")
            st.markdown(response)

# Optional: Show loaded PDFs
with st.expander("Show loaded PDF files"):
    for pdf in pdfs:
        st.write(pdf)

# Optional: Show context stats for debugging
with st.expander("Show context debug info"):
    st.write(f"Number of doc chunks: {len(docs)}")
    st.write(f"Total context length: {len(context)} characters")
    if docs:
        st.write("Sample from first doc chunk:")
        st.write(docs[0].page_content[:300])
