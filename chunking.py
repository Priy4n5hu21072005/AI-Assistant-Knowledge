import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path):
    """
    Load PDF file
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def split_text(documents, chunk_size=300, chunk_overlap=50):
    """
    Split documents into chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def main():

    # Current directory
    current_dir = os.getcwd()

    # PDF path
    pdf_path = os.path.join(current_dir, "sample_ai_notes.pdf")

    print("PDF path:", pdf_path)

    # Check if file exists
    if not os.path.exists(pdf_path):
        print("❌ ERROR: PDF file not found!")
        print("Please place 'sample_ai_notes.pdf' in this folder:")
        print(current_dir)
        return

    print("\nLoading PDF...")

    documents = load_pdf(pdf_path)

    print("Splitting into chunks...")

    chunks = split_text(documents)

    # Combine text
    total_text = "".join([doc.page_content for doc in documents])

    print("\nTotal characters:", len(total_text))
    print("Total chunks:", len(chunks))

    print("\nChunk Preview:\n")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}")
        print("Length:", len(chunk.page_content))
        print(chunk.page_content[:200])  # preview first 200 characters
        print("-" * 50)


if __name__ == "__main__":
    main()