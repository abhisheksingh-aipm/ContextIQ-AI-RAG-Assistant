from pypdf import PdfReader
print("Loading THIS RAG.py")

import chromadb
import hashlib
from google import genai
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import os

load_dotenv()

db = chromadb.PersistentClient(path="chroma_db")
collection = db.get_or_create_collection(name="documents")

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("RAG Engine Loaded Successfully")


# =====================================
# 1. Process PDF
# =====================================

def process_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    full_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            full_text += text

    return full_text


# =====================================
# 2. Generate Document Hash
# =====================================

def generate_document_hash(uploaded_file):

    uploaded_file.seek(0)

    file_bytes = uploaded_file.read()

    uploaded_file.seek(0)

    document_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()

    return document_hash


# =====================================
# 3. Check Document Exists
# =====================================

def document_exists(document_hash):

    results = collection.get()

    metadatas = results.get(
        "metadatas",
        []
    )

    for metadata in metadatas:

        if metadata is None:
            continue

        if metadata.get(
            "document_hash"
        ) == document_hash:

            return True

    return False


# =====================================
# 4. Chunk Text
# =====================================

def chunk_text(full_text):

    chunk_size = 500

    overlap = 100

    chunks = []

    start = 0

    while start < len(full_text):

        end = start + chunk_size

        chunk = full_text[start:end]

        if chunk.strip():

            chunks.append(chunk)

        start = end - overlap

    return chunks


# =====================================
# 5. Create Embeddings
# =====================================

def create_embeddings(chunks):

    print("Creating embeddings...")

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    print(
        "Embeddings created successfully!"
    )

    return embeddings


# =====================================
# 6. Store Embeddings
# =====================================

def store_embeddings(
    document_hash,
    document_name,
    chunks,
    embeddings
):

    if not chunks:

        print(
            "No text found in document."
        )

        return

    ids = []

    metadatas = []

    for i in range(len(chunks)):

        ids.append(
            f"{document_hash}_{i}"
        )

        metadatas.append(
            {
                "document_hash": document_hash,
                "document_name": document_name,
                "chunk_number": i
            }
        )

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(
        "Embeddings stored successfully!"
    )


# =====================================
# 7. Search Selected Documents
# =====================================

def search_documents(
    question,
    selected_documents=None,
    n_results=5
):

    question_embedding = embedding_model.encode(
        question,
        convert_to_numpy=True
    )

    # ---------------------------------
    # If specific documents are selected,
    # search ONLY inside those documents.
    # ---------------------------------

    if selected_documents:

        results = collection.query(
            query_embeddings=[
                question_embedding.tolist()
            ],
            n_results=n_results,
            where={
                "document_name": {
                    "$in": selected_documents
                }
            },
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    else:

        results = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    return results


# =====================================
# 8. Rerank Results
# =====================================

def rerank_results(
    question,
    results
):

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    if not documents:

        return []

    pairs = []

    for document in documents:

        pairs.append(
            (
                question,
                document
            )
        )

    scores = reranker.predict(
        pairs
    )

    ranked_results = []

    for score, document, metadata in zip(
        scores,
        documents,
        metadatas
    ):

        ranked_results.append(
            (
                score,
                document,
                metadata
            )
        )

    ranked_results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return ranked_results


# =====================================
# 9. Generate Answer
# =====================================

def generate_answer(
    question,
    ranked_results
):

    if not ranked_results:

        return (
            "I couldn't find relevant information "
            "in the selected documents."
        )

    top_chunks = []

    for score, document, metadata in ranked_results[:5]:

        document_name = metadata[
            "document_name"
        ]

        top_chunks.append(
            f"""
DOCUMENT: {document_name}

CONTENT:
{document}
"""
        )

    context = "\n\n".join(
        top_chunks
    )

    prompt = f"""
You are ContextIQ, an AI document assistant.

Your job is to answer ONLY using the documents
provided in the context below.

IMPORTANT RULES:

1. Use only information from the provided documents.

2. Do not use your general knowledge to answer.

3. Do not invent information.

4. If the answer is not present in the selected
documents, say:

"I couldn't find the requested information in the selected documents."

5. If the user asks for a comparison, compare only
the documents provided in the context.

6. Clearly identify the document names when useful.

7. Never use information from an unrelated document.

8. If there is insufficient information, say so
instead of guessing.

SELECTED DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    return response.text


# =====================================
# 10. Ask Question
# =====================================

conversation_history = []


def ask_question(
    question,
    selected_documents=None
):

    results = search_documents(
        question,
        selected_documents=selected_documents
    )

    ranked_results = rerank_results(
        question,
        results
    )

    answer = generate_answer(
        question,
        ranked_results
    )

    conversation_history.append(
        {
            "question": question,
            "answer": answer
        }
    )

    return (
        answer,
        ranked_results[:3]
    )


# =====================================
# 11. Get Uploaded Documents
# =====================================

def get_uploaded_documents():

    results = collection.get()

    metadatas = results.get(
        "metadatas",
        []
    )

    documents = {}

    for metadata in metadatas:

        if metadata is None:
            continue

        doc_hash = metadata[
            "document_hash"
        ]

        if doc_hash not in documents:

            documents[doc_hash] = {
                "name": metadata[
                    "document_name"
                ],
                "chunks": 0
            }

        documents[doc_hash][
            "chunks"
        ] += 1

    return documents
