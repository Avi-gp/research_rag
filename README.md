🧠 ResearchMind

RAG-Powered Research Paper Q&A and Summarization System

📌 Overview

ResearchMind is a Streamlit-based application that leverages RAG (Retrieval-Augmented Generation) and a FastAPI backend to help researchers:

📤 Upload and process research papers (PDFs)

❓ Ask AI-powered questions with source-based answers

📋 Generate research summaries (executive, comprehensive, methodology-focused, etc.)

📊 View system statistics and embeddings information

🗂 Manage stored PDFs and vector databases

🚀 Getting Started
Prerequisites

Python 3.9+

FastAPI server running (python api/main.py)

Install dependencies:

pip install -r requirements.txt

Run the App

Start the FastAPI backend:

python api/main.py


Launch the Streamlit app:

streamlit run streamlit_app.py

🔑 Features

Upload multiple PDF research papers

RAG-based Q&A with similarity threshold tuning

Smart research summarization with focus topics

Interactive statistics and system health monitoring

Database reset and management options

📂 Project Structure
├── api/               # FastAPI backend
├── config/            # Settings and configurations
├── data/              # Stored PDFs and vector database
├── streamlit_app.py   # Streamlit frontend
└── requirements.txt   # Dependencies

⚠️ Notes

Ensure FastAPI server is running before using the Streamlit UI

Only PDF files are supported for ingestion

Database management actions (resets, clears) are irreversible

Would you like me to also create a very concise version (3–4 sections only) for GitHub, or keep this medium-length one?

You said:
add clone repo code and others important content too
ChatGPT said:

Got it 👍
Here’s an improved README.md with clone instructions and the important content added:

🧠 ResearchMind

RAG-Powered Research Paper Q&A and Summarization System

📌 Overview

ResearchMind is a modern Streamlit-based application with a FastAPI backend that enables researchers to:

📤 Upload and process research papers (PDFs)

❓ Ask AI-powered questions with citations & similarity scores

📋 Generate summaries (comprehensive, executive, methodology-focused, etc.)

📊 View system statistics and embeddings information

🗂 Manage stored PDFs and vector databases

🚀 Getting Started
1️⃣ Clone the Repository
git clone https://github.com/your-username/researchmind.git
cd researchmind

2️⃣ Install Dependencies

Create a virtual environment (recommended) and install required packages:

pip install -r requirements.txt

3️⃣ Start the Backend (FastAPI)
python api/main.py

4️⃣ Run the Frontend (Streamlit)

In a new terminal:

streamlit run streamlit_app.py

🔑 Features

✅ Upload PDFs – batch upload and ingestion

✅ RAG-based Q&A – ask questions, tune similarity threshold

✅ Summarization – multiple summary types (executive, key findings, methodology, etc.)

✅ System Statistics – embeddings, storage usage, search method details

✅ Database Management – clear PDFs, vector store, or reset everything

📂 Project Structure
├── api/               # FastAPI backend
├── config/            # Settings and configurations
├── data/              # Stored PDFs and vector database
├── streamlit_app.py   # Streamlit frontend
└── requirements.txt   # Dependencies

⚠️ Notes

Make sure the FastAPI server is running before starting Streamlit.

Only PDF files are supported for ingestion.

Database reset/clear actions are permanent.
