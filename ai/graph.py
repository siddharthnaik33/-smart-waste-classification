
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


# =========================================================
# Embedding Model
# =========================================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# =========================================================
# ChromaDB
# =========================================================

vectorstore = Chroma(
    persist_directory=str(VECTORSTORE_DIR),
    embedding_function=embeddings
)


# =========================================================
# LLM
# =========================================================

llm = ChatOllama(
    model="qwen2.5:0.5b",
    temperature=0
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

    # Retrieve ONLY the document matching
    # the predicted waste category.
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
# Node 3: Generate AI Response
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

Answer:
"""

    print("[LLM] Generating response...")

    result = llm.invoke(prompt)

    state["response"] = result.content

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


# Add nodes
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


# Entry point
graph_builder.set_entry_point(
    "confidence_check"
)


# Conditional routing
graph_builder.add_conditional_edges(
    "confidence_check",
    route_after_confidence,
    {
        "finish": END,
        "retrieve": "retrieve"
    }
)


# RAG → LLM
graph_builder.add_edge(
    "retrieve",
    "generate"
)


# LLM → END
graph_builder.add_edge(
    "generate",
    END
)


# Compile graph
graph = graph_builder.compile()


# =========================================================
# Test
# =========================================================

if __name__ == "__main__":

    print("\n========== WASTE AI AGENT TEST ==========\n")

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