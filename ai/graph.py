from pathlib import Path
from typing import TypedDict
import os

from google import genai
from google.genai import types

from langgraph.graph import StateGraph, END
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


# =========================================================
# Environment
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

VECTORSTORE_DIR = BASE_DIR / "vectorstore"


# =========================================================
# Gemini Embeddings
# =========================================================

class GeminiEmbeddings(Embeddings):

    def embed_documents(self, texts):

        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768
            )
        )

        return [
            embedding.values
            for embedding in result.embeddings
        ]

    def embed_query(self, text):

        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768
            )
        )

        return result.embeddings[0].values


embeddings = GeminiEmbeddings()


# =========================================================
# ChromaDB
# =========================================================

vectorstore = Chroma(
    persist_directory=str(VECTORSTORE_DIR),
    embedding_function=embeddings
)


# =========================================================
# LangGraph State
# =========================================================

class WasteState(TypedDict):
    waste_class: str
    confidence: float
    context: str
    response: str


# =========================================================
# Node 1: Confidence Check
# =========================================================

def check_confidence(state: WasteState):

    confidence = state["confidence"]

    print(
        f"\n[Confidence Check] "
        f"{confidence:.2f}%"
    )

    if confidence < 60:

        state["response"] = (
            f"I identified this image as "
            f"{state['waste_class']} with "
            f"{confidence:.2f}% confidence.\n\n"
            "I'm not completely confident about this "
            "classification. Would you like to upload "
            "a clearer image?"
        )

    return state


# =========================================================
# Node 2: Retrieve Category-Specific Information
# =========================================================

def retrieve_information(state: WasteState):

    waste_class = state["waste_class"].strip()

    print(
        f"[RAG] Retrieving information for: "
        f"{waste_class}"
    )

    documents = vectorstore.similarity_search(
        query=waste_class,
        k=1,
        filter={
            "category": waste_class
        }
    )

    if documents:

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        print(
            f"[RAG] Retrieved category: "
            f"{documents[0].metadata.get('category')}"
        )

    else:

        context = (
            f"No specific recycling information "
            f"was found for {waste_class}."
        )

        print(
            "[RAG] No matching category found."
        )

    state["context"] = context

    return state


# =========================================================
# Node 3: Generate AI Response using Gemini
# =========================================================

def generate_response(state: WasteState):

    prompt = f"""
You are an AI waste recycling assistant.

The image classifier identified the following waste:

Waste category: {state["waste_class"]}
Confidence: {state["confidence"]:.2f}%

IMPORTANT RULES:

- Use ONLY the recycling information provided below.
- Do NOT discuss other waste categories.
- Do NOT invent recycling rules.
- Do NOT claim that an item is universally recyclable.
- Local recycling rules may differ.

Recycling information:

{state["context"]}

Your task:

1. Tell the user what was detected.
2. Mention the confidence percentage.
3. Give a short practical recycling or disposal recommendation.
4. Ask exactly ONE useful follow-up question.
5. Keep the answer concise and conversational.

Use this format:

Detected:
<category and confidence>

Recommendation:
<practical recommendation>

Follow-up question:
<one relevant question>
"""

    print("[LLM] Generating response using Gemini...")

    result = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=300
        )
    )

    state["response"] = result.text

    return state


# =========================================================
# Routing
# =========================================================

def route_after_confidence(state: WasteState):

    if state["confidence"] < 60:
        return "finish"

    return "retrieve"


# =========================================================
# Build LangGraph
# =========================================================

graph_builder = StateGraph(WasteState)


graph_builder.add_node(
    "confidence_check",
    check_confidence
)

graph_builder.add_node(
    "retrieve",
    retrieve_information
)

graph_builder.add_node(
    "generate",
    generate_response
)


graph_builder.set_entry_point(
    "confidence_check"
)


graph_builder.add_conditional_edges(
    "confidence_check",
    route_after_confidence,
    {
        "finish": END,
        "retrieve": "retrieve"
    }
)


graph_builder.add_edge(
    "retrieve",
    "generate"
)


graph_builder.add_edge(
    "generate",
    END
)


graph = graph_builder.compile()


# =========================================================
# Local Test
# =========================================================

if __name__ == "__main__":

    print(
        "\n========== WASTE AI AGENT TEST ==========\n"
    )

    waste_class = input(
        "Enter waste class: "
    ).strip()

    confidence = float(
        input("Enter confidence: ")
    )

    result = graph.invoke({
        "waste_class": waste_class,
        "confidence": confidence,
        "context": "",
        "response": ""
    })

    print(
        "\n========== AI AGENT RESPONSE ==========\n"
    )

    print(result["response"])