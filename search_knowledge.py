
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load the same embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Open the existing local database
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="abdullah_ai_knowledge",
)

# Ask a question
question = "Who founded Abdullah AI?"

# Retrieve the most relevant chunks
results = db.similarity_search(question, k=2)

print("\nQuestion:", question)
print("\nRetrieved information:")

for i, document in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(document.page_content)
    print("Source:", document.metadata.get("source"))
