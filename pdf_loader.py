import os
from pypdf import PdfReader

base_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(base_dir, "data", "sample_ai_notes.pdf")

print("PDF Path:", pdf_path)

reader = PdfReader(pdf_path)

print("Total pages:", len(reader.pages))

text = ""

for i, page in enumerate(reader.pages):
    extracted = page.extract_text()
    print(f"Page {i+1} text:", extracted)

    if extracted:
        text += extracted

print("\nTotal text length:", len(text))
print("\nPreview:\n", text[:500])