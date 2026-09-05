from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# Path to ChromaDB
BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


# Same embedding model used during ingestion
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# Load existing ChromaDB
vectorstore = Chroma(
    persist_directory=str(VECTORSTORE_DIR),
    embedding_function=embeddings
)


# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# Test query
query = input("Ask a recycling question: ")

documents = retriever.invoke(query)


print("\n===== RETRIEVED INFORMATION =====\n")

for i, document in enumerate(documents, start=1):
    print(f"--- Result {i} ---")
    print(document.page_content)
    print()