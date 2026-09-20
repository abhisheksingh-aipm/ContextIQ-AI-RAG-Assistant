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
# Session State Initialization
# =====================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Key attached directly to multiselect widget
if "selected_doc_names" not in st.session_state:
    st.session_state.selected_doc_names = []

# Tracks previously uploaded file names to detect NEW uploads
if "previous_upload_names" not in st.session_state:
    st.session_state.previous_upload_names = []


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


# =====================================
# Process Uploads & Update Multiselect
# =====================================
if uploaded_files:
    current_upload_names = [f.name for f in uploaded_files]

    # Detect if NEW files were uploaded in this cycle
    newly_added_files = [
        name for name in current_upload_names 
        if name not in st.session_state.previous_upload_names
    ]

    # Process each uploaded file into ChromaDB
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
                st.error("No readable text found in this PDF.")
                continue

            embeddings = create_embeddings(chunks)
            store_embeddings(
                document_hash,
                uploaded_file.name,
                chunks,
                embeddings
            )
            st.success("Document indexed successfully.")

    # IF NEW FILES WERE ADDED: Automatically select ONLY the newly uploaded files
    if newly_added_files:
        st.session_state.selected_doc_names = current_upload_names
        st.session_state.previous_upload_names = current_upload_names
        st.rerun()


# =====================================
# Sidebar: History & Manual Selection
# =====================================
st.sidebar.title("AI Knowledge Base")

documents = get_uploaded_documents()
st.sidebar.metric("Total Indexed Documents", len(documents))

for doc in documents.values():
    st.sidebar.write(f"📄 {doc['name']}")
    st.sidebar.caption(f"{doc['chunks']} Chunks")

st.sidebar.divider()

all_document_names = [doc["name"] for doc in documents.values()]

# Clean state to ensure only existing documents are in the widget state
st.session_state.selected_doc_names = [
    name for name in st.session_state.selected_doc_names 
    if name in all_document_names
]

# Multiselect widget bound directly to st.session_state.selected_doc_names via key
selected_documents = st.sidebar.multiselect(
    "Select documents to use from history",
    options=all_document_names,
    key="selected_doc_names"
)


# =====================================
# Ask Question
# =====================================
st.divider()
st.subheader("💬 Ask a Question")

question = st.text_input("Enter your question")

if question:
    if not selected_documents:
        st.warning("Please select or upload at least one document before asking a question.")
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

        # Display Sources
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

        # Display History
        st.divider()
        st.subheader("💬 Chat History")

        for chat in st.session_state.chat_history:
            st.chat_message("user").write(chat["question"])
            st.chat_message("assistant").write(chat["answer"])

elif not documents:
    st.info("👆 Upload one or more PDFs to get started.")
