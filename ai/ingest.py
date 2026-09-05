from pathlib import Path
import os

from google import genai
from google.genai import types

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = BASE_DIR / "knowledge" / "recycling_guide.txt"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )


# --------------------------------------------------
# Gemini Client
# --------------------------------------------------

client = genai.Client(api_key=GEMINI_API_KEY)


# --------------------------------------------------
# Gemini Embeddings Wrapper
# --------------------------------------------------

class GeminiEmbeddings(Embeddings):

    def embed_documents(self, texts):
        embeddings = []

        for text in texts:
            response = client.models.embed_content(
                model="gemini-embedding-001",
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=768
                )
            )

            embeddings.append(response.embeddings[0].values)

        return embeddings

    def embed_query(self, text):
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768
            )
        )

        return response.embeddings[0].values


# --------------------------------------------------
# Load Knowledge File
# --------------------------------------------------

with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
    text = f.read()


# --------------------------------------------------
# Waste Categories
# --------------------------------------------------

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


# --------------------------------------------------
# Create One Document Per Category
# --------------------------------------------------

category_documents = []

for category in categories:

    start_marker = f"{category}:"

    start_index = text.find(start_marker)

    if start_index == -1:
        print(f"Warning: {category} not found")
        continue

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


# --------------------------------------------------
# Gemini Embedding Model
# --------------------------------------------------

embeddings = GeminiEmbeddings()


# --------------------------------------------------
# Create ChromaDB
# --------------------------------------------------

vectorstore = Chroma.from_documents(
    documents=category_documents,
    embedding=embeddings,
    persist_directory=str(VECTORSTORE_DIR)
)


print("ChromaDB vector store created successfully!")
print(f"Saved at: {VECTORSTORE_DIR}")
print("Embedding model: gemini-embedding-001")
print("Embedding dimension: 768")