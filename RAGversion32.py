from pypdf import PdfReader
print("Loading THIS RAG.py")

import chromadb
import hashlib
from google import genai
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

# ==========================
# ChromaDB
# ==========================

db = chromadb.PersistentClient(path="chroma_db")

collection = db.get_or_create_collection(
    name="documents"
)

# ==========================
# Gemini Client
# ==========================

import os

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# ==========================
# Cross Encoder
# ==========================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("RAG Engine Loaded Successfully")


# ==========================
# Function 1
# Read PDF
# ==========================

def process_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    full_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:

            full_text += text

    return full_text


# ==========================
# Function 2
# Generate Document Hash
# ==========================

def generate_document_hash(uploaded_file):

    uploaded_file.seek(0)

    file_bytes = uploaded_file.read()

    uploaded_file.seek(0)

    document_hash = hashlib.sha256(file_bytes).hexdigest()

    return document_hash


# ==========================
# Function 3
# Check Existing Document
# ==========================

def document_exists(document_hash):

    results = collection.get()

    metadatas = results.get("metadatas", [])

    for metadata in metadatas:

        if metadata is None:
            continue

        if metadata.get("document_hash") == document_hash:

            return True

    return False


# ==========================
# Function 4
# Chunk Text
# ==========================

def chunk_text(full_text):

    chunk_size = 500

    overlap = 100

    chunks = []

    start = 0

    while start < len(full_text):

        end = start + chunk_size

        chunk = full_text[start:end]

        chunks.append(chunk)

        start = end - overlap

    return chunks


# ==========================
# Function 5
# Create Embeddings
# ==========================

def create_embeddings(chunks):

    print("Creating embeddings...")

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    print("Embeddings created successfully!")

    return embeddings


# ==========================
# Function 6
# Store Embeddings
# ==========================

def store_embeddings(
    document_hash,
    document_name,
    chunks,
    embeddings
):

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

    print("Embeddings stored successfully!")


# ==========================
# Function 7
# Search Selected Documents
# ==========================

def search_documents(question, selected_documents=None, n_results=5):

    question_embedding = embedding_model.encode(
        question,
        convert_to_numpy=True
    )

    results = collection.query(
        query_embeddings=[
            question_embedding.tolist()
        ],
        n_results=50,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    if selected_documents:

        filtered_documents = []
        filtered_metadatas = []
        filtered_distances = []

        for document, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):

            if metadata["document_name"] in selected_documents:

                filtered_documents.append(document)
                filtered_metadatas.append(metadata)
                filtered_distances.append(distance)

        results = {
            "documents": [
                filtered_documents[:n_results]
            ],
            "metadatas": [
                filtered_metadatas[:n_results]
            ],
            "distances": [
                filtered_distances[:n_results]
            ]
        }

    return results


# ==========================
# Function 8
# Rerank Search Results
# ==========================

def rerank_results(question, results):

    documents = results["documents"][0]

    metadatas = results["metadatas"][0]

    pairs = []

    for document in documents:

        pairs.append(
            (question, document)
        )

    scores = reranker.predict(pairs)

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

# ==========================
# Function 9
# Generate Answer
# ==========================

def generate_answer(question, ranked_results):

    top_chunks = []

    for score, document, metadata in ranked_results[:3]:

        top_chunks.append(document)

    context = "\n\n".join(top_chunks)

    prompt = f"""
You are an AI Assistant.

Answer ONLY from the context below.

If the answer is not present, say:

"I couldn't find the answer in the uploaded document."

Context:

{context}

Question:

{question}

Answer:
"""

    response = client.models.generate_content(

        model="gemini-3-flash-preview",

        contents=prompt

    )

    return response.text

# ==========================
# Function 10
# Ask Question
# ==========================

conversation_history = []


def ask_question(question):

    results = search_documents(question)

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

    return answer, ranked_results[:3]

# ==========================
# Function 12
# Get Uploaded Documents
# ==========================

def get_uploaded_documents():

    results = collection.get()

    metadatas = results.get("metadatas", [])

    documents = {}

    for metadata in metadatas:

        if metadata is None:
            continue

        doc_hash = metadata["document_hash"]

        if doc_hash not in documents:

            documents[doc_hash] = {
                "name": metadata["document_name"],
                "chunks": 0
            }

        documents[doc_hash]["chunks"] += 1

    return documents
