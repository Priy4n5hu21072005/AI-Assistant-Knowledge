# 🧠 AI Knowledge Assistant 

<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/FAISS-0052CC?style=for-the-badge&logo=facebook&logoColor=white" alt="FAISS" />
  <img src="https://img.shields.io/badge/HuggingFace-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama" />
</div>

<br/>

## 📖 Overview

### The Problem
In the modern workplace and academic environments, we suffer from "information overload." Extracting specific, accurate answers, summaries, or insights from lengthy PDF documents, research papers, and corporate manuals is highly time-consuming. Traditional `Ctrl+F` based keyword searches fail miserably at understanding context, leading to inaccurate knowledge retrieval and reduced productivity. 

### The Solution
The **AI Knowledge Assistant** solves this problem by serving as a localized, intelligent document search engine utilizing a **Retrieval-Augmented Generation (RAG)** architecture. It reads your PDF documents, contextually chunks the information, stores it in a high-speed vector database (FAISS), and allows you to chat with your documents naturally using Advanced AI. It bridges the gap between raw unstructured files and actionable insights.

---

## ✨ Key Features
- **📄 Automated Document Ingestion:** Intelligent extraction of text from complex PDFs.
- **✂️ Context-Aware Chunking:** Utilizes recursive text splitting to maintain contextual boundaries.
- **🔍 Semantic Search:** Blazing fast mathematical similarity search through the FAISS Vector Database.
- **🔐 Secure & Local Inference:** Capable of running offline local LLMs (via Ollama) ensuring privacy for sensitive files.
- **🤖 Modular RAG Pipeline:** Extensible components for extraction, embedding, retrieval, and text generation.

---

## 🛠️ Tech Stack
- **Core Language:** Python
- **Orchestration:** LangChain
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Store:** FAISS 
- **LLM Engine:** Local LLMs via Ollama
- **Document Parsing:** PyPDF
- **Frontend:** Streamlit 

---

## 📂 Professional Project Structure

```text
📦 AI_Knowlwdge_Assistant
 ┣ 📂 data/                   # Directory for storing PDFs and raw data
 ┃ ┗ 📜 sample_ai_notes.pdf   # Sample knowledge document
 ┣ 📂 src/                    # Main source code directory
 ┃ ┣ 📜 pdf_loader.py         # PDF processing and text extraction
 ┃ ┣ 📜 chunking.py           # Text splitting and semantic chunking logic
 ┃ ┣ 📜 vector_store.py       # Embedding generation and FAISS DB setup
 ┃ ┗ 📜 rag_pipeline.py       # Core RAG orchestration logic
 ┣ 📜 app.py                  # Main Application Entrypoint UI (Streamlit/FastAPI)
 ┣ 📜 requirements.txt        # Virtual Environment dependencies
 ┗ 📜 README.md               # Project documentation
```

---

## ⚙️ Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/Priy4n5hu21072005/AI-Assistant-Knowledge.git
cd AI_Knowlwdge_Assistant
```

**2. Set up virtual environment**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run Vector Engine & API**
Place your target PDFs into the `data/` directory and build your knowledge embeddings.
```bash
python src/vector_store.py
```

**5. Launch the Application**
```bash
streamlit run app.py
```

---

## 🚧 Future Scope Roadmap
- [ ] Implement multi-document retrieval pipeline.
- [ ] Add chat-memory to the Streamlit context window.
- [ ] Deploy utilizing Docker and cloud infrastructure.

> **Note:** The RAG pipeline relies on system memory. Adjust the chunk sizes and overlapping based on your hardware constraints.