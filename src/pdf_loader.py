"""
PDF Loader Module
-----------------
Handles raw text extraction from PDF documents using PyPDF.
"""

import os
from pypdf import PdfReader


# Resolve paths relative to project root (one level up from src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_pdf_text(file_path: str) -> str:
    """
    Extract and return all text from a PDF file.

    Args:
        file_path: Absolute or relative path to the PDF.

    Returns:
        Concatenated text from every page of the PDF.
    """
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    return text


def main():
    pdf_path = os.path.join(DATA_DIR, "sample_ai_notes.pdf")
    print("PDF Path:", pdf_path)

    if not os.path.exists(pdf_path):
        print("❌ ERROR: PDF file not found!")
        print(f"Please place your PDF in: {DATA_DIR}")
        return

    text = load_pdf_text(pdf_path)

    print(f"\nTotal text length: {len(text)}")
    print(f"\nPreview:\n{text[:500]}")


if __name__ == "__main__":
    main()