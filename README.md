# RAG AI Assistant

A local Retrieval-Augmented Generation (RAG) assistant built with Python, LangChain, Streamlit, and Ollama.

## Features

- Load and search a local knowledge base.
- Retrieve relevant information using vector embeddings.
- Generate answers using a language model.
- Interact through a Streamlit web interface.
- Keep uploaded documents and the local vector database on the user's machine.

## Tech Stack

- Python
- Streamlit
- LangChain
- Ollama
- ChromaDB
- Hugging Face Sentence Transformers

## Getting Started

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd RAG_AI_Assistant
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama

Install Ollama and download a compatible local language model. Make sure the Ollama service is running.

### 5. Run the application

```bash
streamlit run app.py
```

## Privacy

The local vector database and uploaded documents are excluded from Git using `.gitignore`.

Do not commit API keys, `.env` files, credentials, or private documents.

## Project Status

Proof of concept / active development.
