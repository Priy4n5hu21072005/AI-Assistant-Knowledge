"""
Vector Store Module
-------------------
Generates sentence embeddings using HuggingFace models and
indexes them in a FAISS vector database for similarity search.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# Resolve paths relative to project root (one level up from src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_pdf(file_path: str):
    """Load a PDF file into LangChain Document objects."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_text(documents, chunk_size: int = 300, chunk_overlap: int = 50):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def create_vector_db(chunks, model_name: str = "all-MiniLM-L6-v2"):
    """
    Create a FAISS vector store from document chunks.

    Args:
        chunks:     List of LangChain Document chunks.
        model_name: HuggingFace embedding model identifier.

    Returns:
        FAISS vector store instance.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def main():
    pdf_path = os.path.join(DATA_DIR, "sample_ai_notes.pdf")

    if not os.path.exists(pdf_path):
        print("❌ ERROR: PDF file not found!")
        print(f"Please place your PDF in: {DATA_DIR}")
        return

    print("Loading PDF...")
    documents = load_pdf(pdf_path)

    print("Splitting text...")
    chunks = split_text(documents)

    print("Creating embeddings & vector database...")
    vectorstore = create_vector_db(chunks)

    print(f"\n✅ Vector database created successfully!")
    print(f"Total chunks stored: {len(chunks)}")

    # 🔎 Similarity Search Test
    query = "What is Artificial Intelligence?"
    results = vectorstore.similarity_search(query, k=2)

    print("\nSearch Results:\n")
    for res in results:
        print(res.page_content)
        print("------")


if __name__ == "__main__":
    main()