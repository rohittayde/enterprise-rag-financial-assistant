import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Dynamically get the absolute path to the folder containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Lock in the exact paths using the base directory
pdf_path = os.path.join(BASE_DIR, "financial_report.pdf")
db_directory = os.path.join(BASE_DIR, "chroma_db")

print(f"Looking for PDF at: {pdf_path}")

# Load the PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split the text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    length_function=len
)
chunks = text_splitter.split_documents(documents)

# Create embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Generate and save the vector database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=db_directory
)

print(f"Successfully processed {len(chunks)} chunks.")
print(f"Vector database saved to {db_directory}")
