import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

os.environ["GROQ_API_KEY"] = "YOUR_ACTUAL_GROQ_KEY_HERE"

db_directory = "./chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = Chroma(
    persist_directory=db_directory, 
    embedding_function=embedding_model
)

# Using Llama 3 8B, hosted on Groq's lightning-fast servers
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 10})
)

query = "Look at the Consolidated Statements of Operations. What is the exact Total Net Sales or Revenue amount in dollars for 2025?"
print(f"Asking: {query}\n")

response = qa_chain.invoke(query)

print("--- AI Analyst Answer ---")
print(response['result'])