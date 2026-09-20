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

# =====================================
# Title
# =====================================

st.title("AI Knowledge Assistant")

st.markdown(
    "Search, understand, and chat with your documents using AI-powered semantic search."
)

st.caption(
    "Upload documents and ask questions using AI-powered semantic search."
)

# =====================================
# Sidebar
# =====================================

st.sidebar.title("AI Knowledge Base")

documents = get_uploaded_documents()

st.sidebar.metric(
    "Documents",
    len(documents)
)

for doc in documents.values():

    st.sidebar.write(
        f"📄 {doc['name']}"
    )

    st.sidebar.caption(
        f"{doc['chunks']} Chunks"
    )


# =====================================
# Select Documents
# =====================================

if documents:

    selected_documents = st.sidebar.multiselect(
        "Select documents to use",
        options=[
            doc["name"]
            for doc in get_uploaded_documents().values()
        ]
    )

else:

    selected_documents = []


# =====================================
# Upload PDFs
# =====================================

st.subheader("📄 Upload Documents")

uploaded_files = st.file_uploader(
    "Drag & drop PDF files here or click to browse",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    for uploaded_file in uploaded_files:

        st.divider()

        st.subheader("Uploaded Document")

        st.write(
            f"**{uploaded_file.name}**"
        )

        document_hash = generate_document_hash(
            uploaded_file
        )

        if document_exists(document_hash):

            st.info(
                "Document already indexed and ready for search."
            )

        else:

            st.info(
                "Processing new document..."
            )

            full_text = process_pdf(
                uploaded_file
            )

            chunks = chunk_text(
                full_text
            )

            embeddings = create_embeddings(
                chunks
            )

            store_embeddings(
                document_hash,
                uploaded_file.name,
                chunks,
                embeddings
            )

            st.success(
                "Document indexed Successfully"
            )

    st.divider()

    # =====================================
    # Ask Question
    # =====================================

    question = st.text_input(
        "Ask a Question"
    )

    if question:

        if not selected_documents:

            st.warning(
                "Please select at least one document from the sidebar before asking a question."
            )

        else:

            with st.spinner(
                "🤖 Gemini is thinking..."
            ):

                answer, sources = ask_question(
                    question,
                    selected_documents=selected_documents
                )

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer
                }
            )

            st.success(
                "✅ Answer Generated"
            )

            st.write(
                "## Answer"
            )

            placeholder = st.empty()

            stream_text = ""

            for word in answer.split():

                stream_text += word + " "

                placeholder.markdown(
                    stream_text
                )

                time.sleep(0.02)

            # =====================================
            # Sources
            # =====================================

            st.divider()

            st.subheader(
                "📚 Sources Used"
            )

            for i, (
                score,
                document,
                metadata
            ) in enumerate(sources):

                with st.expander(
                    f"Source {i+1} | Score: {score:.3f}"
                ):

                    st.write(
                        f"📄 Document: {metadata['document_name']}"
                    )

                    st.write(
                        f"📑 Chunk: {metadata['chunk_number']}"
                    )

                    st.write(
                        document
                    )

            # =====================================
            # Chat History
            # =====================================

            st.divider()

            st.subheader(
                "💬 Chat History"
            )

            for chat in st.session_state.chat_history:

                st.chat_message(
                    "user"
                ).write(
                    chat["question"]
                )

                st.chat_message(
                    "assistant"
                ).write(
                    chat["answer"]
                )

else:

    st.info(
        "👆 Please upload one or more PDFs."
    )
