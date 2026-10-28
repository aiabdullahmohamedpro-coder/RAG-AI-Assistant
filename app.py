
import hashlib
from pathlib import Path

import streamlit as st

from pypdf import PdfReader
from io import BytesIO

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==================================================
# 1. Page configuration
# ==================================================

st.set_page_config(
    page_title="Abdullah AI Assistant",
    page_icon="🤖",
    layout="centered",
)


# ==================================================
# 2. Project settings
# ==================================================

COLLECTION_NAME = "abdullah_ai_knowledge"
DATABASE_PATH = "./chroma_db"
UPLOADS_PATH = Path("./uploaded_files")

UPLOADS_PATH.mkdir(exist_ok=True)


# ==================================================
# 3. Load local AI components
# ==================================================

@st.cache_resource
def load_assistant():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory=DATABASE_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    llm = ChatOllama(
        model="second_constantine/qwen3-A3B:30b-IQ4_XS",
        temperature=0,
    )

    rewrite_prompt = ChatPromptTemplate.from_template("""
Rewrite the latest question as a standalone question.

Use the conversation history only to resolve references
such as "it", "they", or "that company".

Do not answer the question.
Do not add facts.
Return only the rewritten question.

Conversation history:
{chat_history}

Latest question:
{question}

Standalone question:
""")

    answer_prompt = ChatPromptTemplate.from_template("""
You are Abdullah AI Assistant.

Answer using only facts supported by the retrieved context.

Rules:
- Keep answers short and direct, usually 1-3 sentences.
- Avoid repeating the same information.
- Do not invent facts or make unsupported assumptions.
- Conversation history helps you understand the question,
  but is not a source of factual evidence.
- If the retrieved context does not contain the answer,
  reply exactly:
  "I could not find this information in the knowledge base."

Conversation history:
{chat_history}

Retrieved context:
{context}

Current question:
{question}

Answer:
""")

    return db, llm, rewrite_prompt, answer_prompt, embeddings


# ==================================================
# 4. Read uploaded files
# ==================================================

def extract_text(uploaded_file):

    filename = uploaded_file.name
    extension = Path(filename).suffix.lower()

    if extension == ".txt":
        text = uploaded_file.getvalue().decode(
            "utf-8-sig",
            errors="replace",
        )

        return text

    if extension == ".pdf":
        pdf_bytes = BytesIO(uploaded_file.getvalue())
        reader = PdfReader(pdf_bytes)

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""

            if page_text.strip():
                pages.append(
                    f"[Page {page_number}]\n{page_text}"
                )

        return "\n\n".join(pages)

    raise ValueError("Only PDF and TXT files are supported.")


# ==================================================
# 5. Split and index a document
# ==================================================

def index_uploaded_file(uploaded_file, db):

    filename = Path(uploaded_file.name).name
    file_bytes = uploaded_file.getvalue()

    # A content hash helps identify identical file contents
    file_hash = hashlib.sha256(file_bytes).hexdigest()

    # Avoid indexing the same content repeatedly
    if file_hash in st.session_state.indexed_hashes:
        return False, "This file has already been indexed in this session."

    text = extract_text(uploaded_file)

    if not text.strip():
        return False, (
            "No readable text was found. If this is a scanned PDF, "
            "OCR will be needed."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )

    chunks = splitter.split_text(text)

    if not chunks:
        return False, "No text chunks were created from this file."

    documents = []

    for chunk_number, chunk in enumerate(chunks, start=1):
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "file_hash": file_hash,
                    "chunk": chunk_number,
                },
            )
        )

    # Save the document chunks to the local vector database
    db.add_documents(documents)

    # Track indexed content during this Streamlit session
    st.session_state.indexed_hashes.add(file_hash)

    # Save an original copy of the uploaded file
    saved_path = UPLOADS_PATH / f"{file_hash}_{filename}"
    saved_path.write_bytes(file_bytes)

    return True, (
        f"Successfully indexed {filename}: "
        f"{len(documents)} chunks added."
    )


# ==================================================
# 6. Display page heading
# ==================================================

st.title("🤖 Abdullah AI Assistant")

st.caption(
    "A local RAG assistant powered by Qwen, "
    "ChromaDB, and local embeddings."
)


# ==================================================
# 7. Load components
# ==================================================

try:
    db, llm, rewrite_prompt, answer_prompt, embeddings = (
        load_assistant()
    )

except Exception as error:
    st.error("Could not load the local AI components.")
    st.code(str(error))

    st.info(
        "Check that Ollama is running, the Qwen model is available, "
        "and the chroma_db folder exists."
    )

    st.stop()


# ==================================================
# 8. Initialize session state
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_hashes" not in st.session_state:
    st.session_state.indexed_hashes = set()


# ==================================================
# 9. Sidebar: settings and file uploads
# ==================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write("**LLM:** Local Qwen")
    st.write("**Vector database:** ChromaDB")
    st.write("**Embeddings:** MiniLM")
    st.write("**Paid API:** Not required")

    st.divider()

    st.subheader("📚 Add knowledge")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button(
        "Index uploaded files",
        use_container_width=True,
    ):

        if not uploaded_files:
            st.warning("Please select at least one file.")

        else:
            for uploaded_file in uploaded_files:

                try:
                    with st.spinner(
                        f"Processing {uploaded_file.name}..."
                    ):
                        success, message = index_uploaded_file(
                            uploaded_file,
                            db,
                        )

                    if success:
                        st.success(message)
                    else:
                        st.warning(message)

                except Exception as error:
                    st.error(
                        f"Could not process {uploaded_file.name}"
                    )
                    st.code(str(error))

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# ==================================================
# 10. Display conversation history
# ==================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(source)


# ==================================================
# 11. Receive a question
# ==================================================

question = st.chat_input(
    "Ask Abdullah AI something..."
)


if question:

    # Save and display the user's message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:
            # Build history without the current question
            previous_messages = st.session_state.messages[:-1]

            # Use the most recent 6 previous messages
            recent_messages = previous_messages[-6:]

            chat_history = "\n".join(
                (
                    "User: "
                    if message["role"] == "user"
                    else "Assistant: "
                )
                + message["content"]
                for message in recent_messages
            )

            # Step A: Clarify follow-up questions
            search_question = question

            if chat_history.strip():

                with st.spinner(
                    "Understanding the conversation..."
                ):

                    rewrite_messages = (
                        rewrite_prompt.format_messages(
                            chat_history=chat_history,
                            question=question,
                        )
                    )

                    rewritten = llm.invoke(rewrite_messages)
                    candidate = rewritten.content.strip()

                    if candidate:
                        search_question = candidate

            # Step B: Retrieve relevant knowledge
            with st.spinner(
                "Searching the knowledge base..."
            ):

                documents = db.similarity_search(
                    search_question,
                    k=3,
                )

            context = "\n\n".join(
                document.page_content
                for document in documents
            )

            # Step C: Generate a grounded answer
            with st.spinner("Generating the answer..."):

                answer_messages = (
                    answer_prompt.format_messages(
                        chat_history=chat_history,
                        context=context,
                        question=search_question,
                    )
                )

                response = llm.invoke(answer_messages)
                answer = response.content.strip()

            # Collect the retrieved source filenames
            sources = sorted(
                {
                    document.metadata.get(
                        "source",
                        "Unknown source",
                    )
                    for document in documents
                }
            )

            st.markdown(answer)

            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.write(source)

            # Save the assistant's response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        except Exception as error:
            st.error("Something went wrong.")
            st.code(str(error))
