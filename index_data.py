
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Read the knowledge base
file_path = Path("company.txt")
text = file_path.read_text(encoding="utf-8")

# 2. Split the text into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=30,
)

chunks = splitter.split_documents([
    Document(page_content=text, metadata={"source": "company.txt"})
])

print(f"Number of chunks: {len(chunks)}")

# 3. Load a local embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store the chunks in a local Chroma database
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="abdullah_ai_knowledge",
)

print("Knowledge base indexed successfully!")
print("Database location: ./chroma_db")
