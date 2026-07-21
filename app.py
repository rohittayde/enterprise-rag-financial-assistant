import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

# 1. UI Setup
st.set_page_config(page_title="Financial Analyst", page_icon="📊")
st.title("📊 Enterprise RAG - Financial Assistant")
st.markdown("Ask the AI Analyst any question about the SEC 10-K report.")
st.divider()

# 2. Cache the Engine (Zero-Database, In-Memory Build)
@st.cache_resource
def load_rag_chain():
    # A. Find the PDF dynamically using absolute paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(BASE_DIR, "financial_report.pdf")
    
    # B. Load and chunk the PDF directly on boot
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # C. Create embeddings and build IN-MEMORY database (No persist_directory!)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(documents=chunks, embedding=embedding_model)
    
    # D. Initialize LLM (Pulls API key securely from Streamlit Secrets)
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant", 
        temperature=0,
        api_key=st.secrets["GROQ_API_KEY"]
    )
    
    # E. Wire up the chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 10})
    )
    return qa_chain

# Show a loading spinner while the database builds in the background
with st.spinner("Initializing AI and reading financial documents..."):
    qa_chain = load_rag_chain()

# 3. Initialize Chat Memory (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display previous chat messages when the app reloads
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. The Chat Input Box
if prompt := st.chat_input("Ask a financial question..."):
    
    # Render user's prompt on the screen immediately
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Save user's prompt to memory
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 6. Generate and Display the AI Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing financial documents..."):
            response = qa_chain.invoke(prompt)
            answer = response['result']
            st.markdown(answer)
            
    # Save the AI's answer to memory
    st.session_state.messages.append({"role": "assistant", "content": answer})
