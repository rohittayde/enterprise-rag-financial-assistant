# 📊 Enterprise Financial RAG Assistant

An end-to-end Retrieval-Augmented Generation (RAG) application designed to act as an AI Financial Analyst. This tool ingests massive, unstructured financial documents (like SEC 10-K reports) and allows users to ask complex questions, receiving synthesized answers grounded entirely in the provided text.

## 🚀 Project Overview

Financial analysts spend countless hours reading 100+ page SEC filings. This project automates the extraction and synthesis of that data. By leveraging a local vector database and a lightning-fast Large Language Model (LLM), this application bypasses traditional keyword search limitations to understand the *semantic meaning* of financial queries.

### Key Features
* **Conversational Memory:** Built with Streamlit Session State, the AI remembers previous questions, allowing for a natural, chat-like exploration of the financial data.
* **Semantic Search:** Converts financial text into dense mathematical vectors, allowing the system to understand concepts rather than just matching exact words.
* **Local, Secure Storage:** Uses ChromaDB to store documents locally, ensuring sensitive financial PDFs never leave your machine during the database creation phase.
* **Ultra-Fast Inference:** Powered by Groq's specialized hardware (LPUs) and the open-source Llama 3.1 model, generating complex financial summaries in milliseconds.

---

## 🛠️ Architecture & Tech Stack

This project was built using a modern, scalable AI stack:

* **Frontend UI:** `Streamlit`
* **AI Orchestration:** `LangChain` (v1.x architecture)
* **Vector Database:** `ChromaDB` (Local)
* **Embeddings Model:** `HuggingFace` (`all-MiniLM-L6-v2`)
* **LLM Engine:** `Groq API` (`llama-3.1-8b-instant`)
* **Document Processing:** `PyPDFLoader`, `RecursiveCharacterTextSplitter`

### The RAG Pipeline Workflow
1. **Ingestion:** The raw SEC 10-K PDF is loaded using PyPDFLoader.
2. **Chunking:** The document is split into optimized 1,000-character blocks with a 200-character overlap. This prevents financial tables or sentences from being cut in half.
3. **Embedding:** Each chunk is passed through a HuggingFace Sentence Transformer to create a mathematical vector representing its meaning.
4. **Storage:** Vectors are saved permanently to a local `chroma_db` directory.
5. **Retrieval:** When a user asks a question, the system searches the database for the Top 10 most relevant chunks (`k=10`).
6. **Generation:** The retrieved financial paragraphs are sent to Llama 3.1, which reads the context and writes a clear, professional answer.

---

## 📁 Project Structure

```text
financial-rag-assistant/
│
├── chroma_db/               # Local vector database (Auto-generated)
├── venv/                    # Python virtual environment (Ignored in Git)
├── app.py                   # Main Streamlit application and UI
├── create_db.py             # Script to chunk PDF and build the database
├── financial_report.pdf     # The source document (e.g., SEC 10-K)
├── requirements.txt         # Project dependencies
├── .gitignore               # Git ignore file for security
└── README.md                # Project documentation 
