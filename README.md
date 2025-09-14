# 🧠 ResearchMind
RAG-Powered Research Paper Q&A and Summarization System

## 📌 Overview
**ResearchMind** is a modern **Streamlit + FastAPI** application that helps researchers:
- 📤 Upload and process research papers (PDFs)  
- ❓ Ask AI-powered questions with citations & similarity scores  
- 📋 Generate smart summaries (executive, key findings, methodology, etc.)  
- 📊 Monitor system statistics and embeddings  
- 🗂 Manage stored PDFs and vector databases  

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone 'https://github.com/Avi-gp/research_rag.git'

cd research_rag
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Add API Key(s)

Create a .env file in the root directory and add your API keys. For example:
```bash
GOOGLE_API_KEY=your_gemini_key_here (For LLM)

NVIDIA_API_KEY=your_nvidia_api_key (FOR Embedding Model)

```

If you want to use other Embedding Model or LLLM make changes in respective files

### 4️⃣ Start the Backend (FastAPI)
```bash
python api/main.py
```

### 5️⃣ Run the Frontend (Streamlit)
```bash
streamlit run ui/app.py
```

### 🔑 Features

✅ Upload multiple research papers (PDFs)

✅ RAG-based Q&A with similarity threshold tuning

✅ Generate focused or comprehensive summaries

✅ View system statistics & embeddings info

✅ Manage database (clear PDFs, vector store, or reset system)

### 📂 Project Structure
```bash
├── api/               # FastAPI backend
├── config/            # Settings and configurations
├── data/              # Stored PDFs and vector database
├── services/          # # Core logic for LLM, PDF processing, vector store, and RAG pipeline
├── ui/app.py          # Streamlit frontend
└── requirements.txt   # Dependencies
```

### ⚠️ Notes

Make sure the FastAPI server is running before using Streamlit.

Only PDF files are supported for ingestion.

Database management actions (resets, clears) are irreversible.


