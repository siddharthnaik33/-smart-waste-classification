from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


# --------------------------------------------------
# Load embeddings
# --------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# --------------------------------------------------
# Load ChromaDB
# --------------------------------------------------

vectorstore = Chroma(
    persist_directory=str(VECTORSTORE_DIR),
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# --------------------------------------------------
# Load LLM
# --------------------------------------------------

llm = ChatOllama(
    model="qwen2.5:0.5b",
    temperature=0
)


# --------------------------------------------------
# RAG function
# --------------------------------------------------

def generate_recycling_advice(question: str) -> str:

    # Retrieve relevant documents
    documents = retriever.invoke(question)

    # Combine retrieved information
    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # Prompt for the LLM
    prompt = f"""
You are a waste-recycling assistant.

Answer the user's question using the provided recycling
information.

Rules:
- Give a concise and practical answer.
- Identify the waste category when possible.
- Explain whether it should generally be recycled,
  composted, reused, donated, or specially disposed.
- Do not claim that something is universally recyclable.
- Mention that local recycling rules can differ.
- If the retrieved information is insufficient, say so.

Retrieved recycling information:
{context}

User question:
{question}

Answer:
"""

    # Generate answer
    response = llm.invoke(prompt)

    return response.content


# --------------------------------------------------
# Test the RAG + LLM system
# --------------------------------------------------

if __name__ == "__main__":

    question = input(
        "Ask a waste-recycling question: "
    )

    answer = generate_recycling_advice(question)

    print("\n===== AI RECYCLING ASSISTANT =====\n")
    print(answer)