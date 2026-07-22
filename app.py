import streamlit as st
import os
import base64
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

# 1. UI Setup
st.set_page_config(page_title="Financial Analyst", page_icon="📊", layout="wide")
st.title("📊 Enterprise RAG - Financial Assistant")
st.markdown("Ask the AI Analyst any question about the Apple inc SEC 10-K report.")
st.divider()

# 2. PDF Viewer in Sidebar
def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # Embed the PDF into an HTML iframe
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

with st.sidebar:
    st.header("📄 Source Document")
    st.markdown("View or download the Apple inc sec 10 k financial report.")
    
    # Locate the PDF dynamically
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(BASE_DIR, "financial_report.pdf")
    
    if os.path.exists(pdf_path):
        # The Download Button
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📥 Download PDF",
                data=pdf_file,
                file_name="financial_report.pdf",
                mime="application/pdf"
            )
            
        # The Embedded Viewer
        with st.expander("👁️ Preview Document"):
            display_pdf(pdf_path)
    else:
        st.warning("The PDF file is currently unavailable.")

# 3. Cache the Engine (Zero-Database, In-Memory Build)
@st.cache_resource
def load_rag_chain():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(BASE_DIR, "financial_report.pdf")
    
    # Load and chunk the PDF directly on boot
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings and build IN-MEMORY database (No persist_directory!)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(documents=chunks, embedding=embedding_model)
    
    # Initialize LLM (Pulls API key securely from Streamlit Secrets)
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant", 
        temperature=0,
        api_key=st.secrets["GROQ_API_KEY"]
    )
    
    # Wire up the chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 10})
    )
    return qa_chain

# Show a loading spinner while the database builds in the background
with st.spinner("Initializing AI and reading financial documents..."):
    qa_chain = load_rag_chain()

# 4. Initialize Chat Memory (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Display previous chat messages when the app reloads
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. The Chat Input Box
if prompt := st.chat_input("Ask a financial question..."):
    
    # Render user's prompt on the screen immediately
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Save user's prompt to memory
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and Display the AI Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing financial documents..."):
            response = qa_chain.invoke(prompt)
            answer = response['result']
            st.markdown(answer)
            
    # Save the AI's answer to memory
    st.session_state.messages.append({"role": "assistant", "content": answer})
