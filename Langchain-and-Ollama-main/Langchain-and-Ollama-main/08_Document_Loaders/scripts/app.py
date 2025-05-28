import streamlit as st
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import (SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate)
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- SETTINGS ---
BASE_URL = "http://localhost:11434"
MODEL = "llama3.2:3b"
PDF_ROOT = r"D:\ML\HKA_Chatbot\Langchain-and-Ollama-main\Langchain-and-Ollama-main\08_Document_Loaders\rag-dataset"

# --- LOAD AND SPLIT ALL LOCAL PDFs ---
@st.cache_data(show_spinner=True)
def load_all_pdfs_and_chunks(pdf_root=PDF_ROOT):
    docs = []
    files_loaded = []
    for root, dirs, files in os.walk(pdf_root):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                try:
                    loader = PyMuPDFLoader(pdf_path)
                    file_docs = loader.load()
                    # Attach file and page metadata
                    for d in file_docs:
                        d.metadata['source'] = file
                    docs.extend(file_docs)
                    files_loaded.append(file)
                except Exception as e:
                    st.warning(f"Could not read {file}: {e}")

    # Split into semantic chunks for better context (best practice!)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    all_chunks = splitter.split_documents(docs)
    return all_chunks, files_loaded

def format_docs(docs):
    # Show which file/page each chunk is from, for reference
    return "\n\n".join(
        [f"--- Source: {d.metadata.get('source','?')} p{d.metadata.get('page', '?')}\n{d.page_content}" for d in docs]
    )

st.title("PDF Chatbot (All Local PDFs)")
st.caption("Automatically uses all PDFs in the rag-dataset/ folder for context.")

project = st.sidebar.selectbox("Select Project", [
    "Question Answering from PDF Document",
    "PDF Document Summarization",
    "Report Generation from PDF Document"
])
words = st.sidebar.slider("Max words for answer/summary", 20, 2000, 200)

docs, files_loaded = load_all_pdfs_and_chunks()
if not docs:
    st.error("No PDFs found in your rag-dataset folder.")
    st.stop()

# Show loaded files
with st.expander("Show loaded PDF files"):
    st.write(files_loaded)

context = format_docs(docs)
llm = ChatOllama(base_url=BASE_URL, model=MODEL)

if project == "Question Answering from PDF Document":
    st.subheader("Ask a Question")
    question = st.text_input("Your question about the PDFs:")
    if st.button("Get Answer") and question:
        system = SystemMessagePromptTemplate.from_template(
            """
            You are a strict AI assistant. ONLY answer using the provided context below.
            If the context does NOT answer the user's question, say exactly "I don't know" and nothing else.
            Do NOT use your own knowledge or provide general advice if the answer is not in the context.
            Do not answer in more than {words} words.
            """
        )
        prompt = HumanMessagePromptTemplate.from_template(
            """### Context:
{context}

### Question:
{question}

### Answer (use only info from context):"""
        )
        template = ChatPromptTemplate([system, prompt])
        chain = template | llm | StrOutputParser()
        response = chain.invoke({'context': context, 'question': question, 'words': words})
        st.markdown(f"**Answer:** {response}")

elif project == "PDF Document Summarization":
    st.subheader("Summarize All PDFs")
    if st.button("Summarize PDFs"):
        system = SystemMessagePromptTemplate.from_template(
            "You are helpful AI assistant who works as document summarizer. You must not hallucinate or provide any false information."
        )
        prompt = HumanMessagePromptTemplate.from_template(
            "Summarize the given context in {words} words.\n### Context:\n{context}\n### Summary:"
        )
        template = ChatPromptTemplate([system, prompt])
        chain = template | llm | StrOutputParser()
        response = chain.invoke({'context': context, 'words': words})
        st.markdown(f"**Summary:**\n{response}")

elif project == "Report Generation from PDF Document":
    st.subheader("Generate a Report from All PDFs")
    if st.button("Generate Report"):
        system = SystemMessagePromptTemplate.from_template(
            "You are helpful AI assistant who creates detailed markdown reports from PDF context. Do not hallucinate or make up information."
        )
        prompt = HumanMessagePromptTemplate.from_template(
            "Provide a detailed report from the provided context. Write answer in Markdown and do not exceed {words} words.\n### Context:\n{context}\n### Report:"
        )
        template = ChatPromptTemplate([system, prompt])
        chain = template | llm | StrOutputParser()
        response = chain.invoke({'context': context, 'words': words})
        st.markdown(response)
