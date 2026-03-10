"""
RAG Pipeline Module
-------------------
Orchestrates the full Retrieval-Augmented Generation workflow:
  1. Load & chunk documents
  2. Build / load the vector store
  3. Retrieve relevant context for a user query
  4. Generate an answer via an LLM
"""

import os
from src.chunking import load_pdf, split_text
from src.vector_store import create_vector_db


# Resolve paths relative to project root (one level up from src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def build_knowledge_base(pdf_path: str):
    """
    End-to-end pipeline: load PDF → chunk → embed → return vector store.

    Args:
        pdf_path: Path to the source PDF document.

    Returns:
        FAISS vector store ready for similarity search.
    """
    documents = load_pdf(pdf_path)
    chunks = split_text(documents)
    vectorstore = create_vector_db(chunks)
    return vectorstore


def retrieve_context(vectorstore, query: str, top_k: int = 3) -> str:
    """
    Retrieve the most relevant document chunks for a query.

    Args:
        vectorstore: FAISS vector store instance.
        query:       User's natural-language question.
        top_k:       Number of chunks to retrieve.

    Returns:
        Concatenated text of the top-k relevant chunks.
    """
    results = vectorstore.similarity_search(query, k=top_k)
    context = "\n\n".join([doc.page_content for doc in results])
    return context


def generate_answer(query: str, context: str) -> str:
    """
    Generate an answer using the retrieved context.

    Args:
        query:   User's question.
        context: Retrieved document text.

    Returns:
        Generated answer string.

    TODO: Integrate with Ollama or any LLM API.
    """
    # Placeholder — replace with actual LLM call
    prompt = (
        f"Answer the following question based on the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    return prompt


def main():
    pdf_path = os.path.join(DATA_DIR, "sample_ai_notes.pdf")

    if not os.path.exists(pdf_path):
        print("❌ PDF not found. Place your file in the data/ directory.")
        return

    print("🔨 Building knowledge base...")
    vectorstore = build_knowledge_base(pdf_path)
    print("✅ Knowledge base ready.\n")

    query = "What is Artificial Intelligence?"
    print(f"❓ Query: {query}")

    context = retrieve_context(vectorstore, query)
    print(f"\n📄 Retrieved Context:\n{context}\n")

    answer = generate_answer(query, context)
    print(f"💡 Generated Prompt / Answer:\n{answer}")


if __name__ == "__main__":
    main()
