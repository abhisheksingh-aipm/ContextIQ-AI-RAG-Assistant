import streamlit as st
import time

from RAGversion32 import (
    process_pdf,
    chunk_text,
    create_embeddings,
    store_embeddings,
    ask_question,
    generate_document_hash,
    document_exists,
    get_uploaded_documents
)

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide"
)

# =====================================
# Session State
# =====================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Tracks documents selected for searching
if "selected_documents" not in st.session_state:
    st.session_state.selected_documents = []


# =====================================
# Title
# =====================================
st.title("AI Knowledge Assistant")
st.markdown("Search, understand, and chat with your documents using AI-powered semantic search.")
st.caption("Upload documents and ask questions using AI-powered semantic search.")


# =====================================
# Upload Documents
# =====================================
st.subheader("📄 Upload Documents")

uploaded_files = st.file_uploader(
    "Drag & drop PDF files here or click to browse",
    type=["pdf"],
    accept_multiple_files=True
)

new_upload_processed = False

# =====================================
# Process New Uploads
# =====================================
if uploaded_files:
    # Set search context ONLY to the newly uploaded files
    st.session_state.selected_documents = [f.name for f in uploaded_files]

    for uploaded_file in uploaded_files:
        st.divider()
        st.subheader("Uploaded Document")
        st.write(f"**{uploaded_file.name}**")

        document_hash = generate_document_hash(uploaded_file)

        if document_exists(document_hash):
            st.info("Document already indexed and ready for search.")
        else:
            st.info("Processing new document...")
            full_text = process_pdf(uploaded_file)
            chunks = chunk_text(full_text)

            if not chunks:
                st.error(
                    "No readable text was found in this PDF. "
                    "This may be a scanned/image-only PDF."
                )
                continue

            embeddings = create_embeddings(chunks)
            store_embeddings(
                document_hash,
                uploaded_file.name,
                chunks,
                embeddings
            )

            st.success("Document indexed successfully.")
            new_upload_processed = True

    # Force UI to rerun so multiselect selects ONLY the newly uploaded file(s)
    if new_upload_processed:
        st.rerun()


# =====================================
# Sidebar: History & Selection
# =====================================
st.sidebar.title("AI Knowledge Base")

# Retrieve indexed documents history
documents = get_uploaded_documents()

st.sidebar.metric("Total Indexed Documents", len(documents))

for doc in documents.values():
    st.sidebar.write(f"📄 {doc['name']}")
    st.sidebar.caption(f"{doc['chunks']} Chunks")

st.sidebar.divider()

# List of all document names in history
all_document_names = [doc["name"] for doc in documents.values()]

# Ensure session state only contains valid names
valid_selections = [
    doc_name for doc_name in st.session_state.selected_documents 
    if doc_name in all_document_names
]

# Sidebar multiselect for picking from history
selected_documents = st.sidebar.multiselect(
    "Select documents to use from history",
    options=all_document_names,
    default=valid_selections
)

# Update state based on manual user selections in sidebar
st.session_state.selected_documents = selected_documents


# =====================================
# Ask Question
# =====================================
st.divider()
st.subheader("💬 Ask a Question")

question = st.text_input("Enter your question")

if question:
    if not selected_documents:
        st.warning(
            "Please select or upload at least one document before asking a question."
        )
    else:
        with st.spinner("🤖 Gemini is thinking..."):
            answer, sources = ask_question(
                question,
                selected_documents=selected_documents
            )

        # Save Chat History
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })

        # Display Answer
        st.success("✅ Answer Generated")
        st.write("## Answer")

        placeholder = st.empty()
        stream_text = ""

        for word in answer.split():
            stream_text += word + " "
            placeholder.markdown(stream_text)
            time.sleep(0.02)

        # Display Sources Used
        st.divider()
        st.subheader("📚 Sources Used")

        if sources:
            for i, (score, document, metadata) in enumerate(sources):
                with st.expander(f"Source {i+1} | Score: {score:.3f}"):
                    st.write(f"📄 Document: {metadata['document_name']}")
                    st.write(f"📑 Chunk: {metadata['chunk_number']}")
                    st.write(document)
        else:
            st.info("No relevant sources were found in the selected documents.")

        # Display Chat History
        st.divider()
        st.subheader("💬 Chat History")

        for chat in st.session_state.chat_history:
            st.chat_message("user").write(chat["question"])
            st.chat_message("assistant").write(chat["answer"])

elif not documents:
    st.info("👆 Upload one or more PDFs to get started.")
