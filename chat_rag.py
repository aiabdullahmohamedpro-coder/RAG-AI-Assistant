
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# 1. Load the embedding model
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Open the local knowledge database
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="abdullah_ai_knowledge",
)

# 3. Load the local language model
llm = ChatOllama(
    model="second_constantine/qwen3-A3B:30b-IQ4_XS",
    temperature=0,
)

# 4. Define the assistant instructions
prompt = ChatPromptTemplate.from_template("""
You are Abdullah AI Assistant.

Answer the user's question using ONLY the information
provided in the context.

If the answer is not available in the context, say:
"I could not find this information in the knowledge base."

Context:
{context}

Question:
{question}

Answer:
""")

print("\n" + "=" * 45)
print("       Abdullah AI Assistant")
print("=" * 45)
print("Ask questions about Abdullah AI.")
print("Type 'exit' to close the assistant.")
print("-" * 45)

# 5. Keep accepting questions
while True:
    question = input("\nYou: ").strip()

    # Stop when the user types exit
    if question.lower() == "exit":
        print("\nAbdullah AI Assistant: Goodbye!")
        break

    # Ignore empty questions
    if not question:
        continue

    try:
        # 6. Retrieve relevant information
        documents = db.similarity_search(question, k=2)

        context = "\n\n".join(
            document.page_content for document in documents
        )

        # 7. Send the question and context to Qwen
        messages = prompt.format_messages(
            context=context,
            question=question,
        )

        response = llm.invoke(messages)

        # 8. Display the answer
        print("\nAbdullah AI Assistant:")
        print(response.content)

    except Exception as error:
        print("\nAn error occurred:")
        print(error)
