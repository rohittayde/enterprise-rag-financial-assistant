from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db_directory = "./chroma_db"

vector_db = Chroma(
    persist_directory=db_directory, 
    embedding_function=embedding_model
)

query = "What was the total revenue or financial highlights?"
print(f"Question: '{query}'\n")

results = vector_db.similarity_search(query, k=3)

for i, chunk in enumerate(results):
    print(f"--- Top Result {i+1} ---")
    print(chunk.page_content)
    print("\n")