import langchain
import langchain_community
import pypdf
import tiktoken
import chromadb
import sentence_transformers

print(f"LangChain: {langchain.__version__}")
print(f"LangChain Community: {langchain_community.__version__}")
print(f"PyPDF: {pypdf.__version__}")
print(f"TikToken: {tiktoken.__version__}")
print(f"ChromaDB: {chromadb.__version__}")
print(f"Sentence Transformers: {sentence_transformers.__version__}")