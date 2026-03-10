"""
Chunking Module
---------------
Splits PDF documents into semantically meaningful chunks
using LangChain's RecursiveCharacterTextSplitter.
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Resolve paths relative to project root (one level up from src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_pdf(file_path: str):
    """
    Load a PDF file into LangChain Document objects.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of LangChain Document objects (one per page).
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def split_text(documents, chunk_size: int = 300, chunk_overlap: int = 50):
    """
    Split documents into overlapping chunks for embedding.

    Args:
        documents:     List of LangChain Document objects.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def main():
    pdf_path = os.path.join(DATA_DIR, "sample_ai_notes.pdf")
    print("PDF path:", pdf_path)

    if not os.path.exists(pdf_path):
        print("❌ ERROR: PDF file not found!")
        print(f"Please place 'sample_ai_notes.pdf' in: {DATA_DIR}")
        return

    print("\nLoading PDF...")
    documents = load_pdf(pdf_path)

    print("Splitting into chunks...")
    chunks = split_text(documents)

    total_text = "".join([doc.page_content for doc in documents])

    print(f"\nTotal characters: {len(total_text)}")
    print(f"Total chunks:     {len(chunks)}")

    print("\nChunk Preview:\n")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}")
        print(f"Length: {len(chunk.page_content)}")
        print(chunk.page_content[:200])
        print("-" * 50)


if __name__ == "__main__":
    main()