import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_pdf(file_path):

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    return documents


def split_text(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    return chunks


def create_vector_db(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


def main():

    pdf_path = "sample_ai_notes.pdf"

    print("Loading PDF...")
    documents = load_pdf(pdf_path)

    print("Splitting text...")
    chunks = split_text(documents)

    print("Creating embeddings & vector database...")
    vectorstore = create_vector_db(chunks)

    print("\nVector database created successfully!")
    print("Total chunks stored:", len(chunks))

    # 🔎 Search Test
    query = "What is Artificial Intelligence?"

    results = vectorstore.similarity_search(query, k=2)

    print("\nSearch Results:\n")

    for res in results:
        print(res.page_content)
        print("------")


if __name__ == "__main__":
    main()