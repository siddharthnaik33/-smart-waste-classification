from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = BASE_DIR / "knowledge" / "recycling_guide.txt"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


# Load the knowledge file
loader = TextLoader(
    str(KNOWLEDGE_FILE),
    encoding="utf-8"
)

documents = loader.load()

text = documents[0].page_content


# Waste categories
categories = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation"
]


# Create one document for each category
category_documents = []

for category in categories:

    start_marker = f"{category}:"

    start_index = text.find(start_marker)

    if start_index == -1:
        print(f"Warning: {category} not found")
        continue

    # Find the next category
    next_indexes = []

    for next_category in categories:

        if next_category == category:
            continue

        index = text.find(
            f"{next_category}:",
            start_index + len(start_marker)
        )

        if index != -1:
            next_indexes.append(index)

    if next_indexes:
        end_index = min(next_indexes)
    else:
        end_index = len(text)

    category_text = text[start_index:end_index].strip()

    category_documents.append(
        Document(
            page_content=category_text,
            metadata={
                "category": category
            }
        )
    )


print(
    f"Created {len(category_documents)} "
    f"category documents"
)


# Embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# Create ChromaDB
vectorstore = Chroma.from_documents(
    documents=category_documents,
    embedding=embeddings,
    persist_directory=str(VECTORSTORE_DIR)
)


print("ChromaDB vector store created successfully!")
print(f"Saved at: {VECTORSTORE_DIR}")