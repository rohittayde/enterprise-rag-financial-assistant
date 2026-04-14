import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

# Set your Groq API Key
os.environ["GROQ_API_KEY"] = "YOUR_ACTUAL_GROQ_KEY_HERE"

# 1. UI Setup
st.set_page_config(page_title="Financial Analyst", page_icon="📊")
st.title("📊 Enterprise RAG - Financial Assistant")
st.markdown("Ask the AI Analyst any question about the SEC 10-K report.")
st.divider()

# 2. Cache the Engine
@st.cache_resource
def load_rag_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory="./chroma_db", 
        embedding_function=embedding_model
    )
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 10})
    )
    return qa_chain

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