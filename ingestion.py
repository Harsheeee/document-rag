"""
ingestion.py — Data Ingestion Pipeline
========================================
Phase 1 of the RAG pipeline.
Loads documents from data/, chunks them, generates embeddings using
HuggingFace all-MiniLM-L6-v2, and uploads the vectors to a Pinecone index.

Usage (Docker):
    docker compose run --rm rag python ingestion.py
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# ── Load environment variables ───────────────────────────────────────────────
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "rag"  # The index you already created on Pinecone


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Load Documents
# ══════════════════════════════════════════════════════════════════════════════

def load_documents(data_path: str = "data/"):
    """
    Load all PDF and TXT files from the specified directory.
    Returns a list of LangChain Document objects with page_content and metadata.
    """
    documents = []

    # Load PDFs
    pdf_loader = DirectoryLoader(
        path=data_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    pdf_docs = pdf_loader.load()
    print(f"  📄 Loaded {len(pdf_docs)} pages from PDF files.")
    documents.extend(pdf_docs)

    # Load TXT files
    txt_loader = DirectoryLoader(
        path=data_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True,
    )
    txt_docs = txt_loader.load()
    print(f"  📝 Loaded {len(txt_docs)} text files.")
    documents.extend(txt_docs)

    print(f"  ✅ Total documents loaded: {len(documents)}")
    return documents


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Chunk Documents
# ══════════════════════════════════════════════════════════════════════════════

def chunk_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Split loaded documents into smaller chunks using RecursiveCharacterTextSplitter.
    Each chunk retains the original metadata (source file, page number, etc.).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  ✅ Split into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap}).")
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Generate Embeddings & Upload to Pinecone
# ══════════════════════════════════════════════════════════════════════════════

def get_embedding_model():
    """
    Initialize the HuggingFace embedding model.
    Uses 'all-MiniLM-L6-v2' which produces 384-dimensional vectors.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    print("  ✅ HuggingFace embedding model loaded (all-MiniLM-L6-v2, 384 dims).")
    return embedding_model


def upload_to_pinecone(chunks, embedding_model):
    """
    Embed all chunks and upload them to the Pinecone vector index.
    Clears existing vectors first to avoid stale duplicates.
    """
    # Connect to Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    old_count = stats.total_vector_count
    print(f"  📊 Pinecone index '{PINECONE_INDEX_NAME}' — current vectors: {old_count}")

    # Clear old vectors to prevent stale/duplicate results
    if old_count > 0:
        print(f"  🗑️  Deleting {old_count} existing vectors ...")
        index.delete(delete_all=True)
        print(f"  ✅ Old vectors deleted.")

    # Upload chunks: embeds each chunk and upserts to Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        index_name=PINECONE_INDEX_NAME,
    )

    # Re-check stats after upload
    stats = index.describe_index_stats()
    print(f"  ✅ Upload complete — total vectors now: {stats.total_vector_count}")
    return vectorstore


# ══════════════════════════════════════════════════════════════════════════════
# Main Ingestion Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  RAG Data Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load
    print("\n📂 Step 1: Loading documents from data/ ...")
    documents = load_documents("data/")
    if not documents:
        print("  ⚠️  No documents found in data/. Add PDFs or TXT files and re-run.")
        return

    # Step 2: Chunk
    print("\n✂️  Step 2: Chunking documents ...")
    chunks = chunk_documents(documents)

    # Inspect a sample
    print("\n--- Sample Chunk ---")
    print(f"Content: {chunks[0].page_content[:200]}...")
    print(f"Metadata: {chunks[0].metadata}")

    # Step 3: Embed & Upload
    print("\n🧠 Step 3: Generating embeddings and uploading to Pinecone ...")
    embedding_model = get_embedding_model()
    upload_to_pinecone(chunks, embedding_model)

    print("\n" + "=" * 60)
    print("  ✅ Ingestion complete! Your documents are now in Pinecone.")
    print("=" * 60)


if __name__ == "__main__":
    main()
