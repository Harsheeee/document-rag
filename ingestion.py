"""
ingestion.py - Data Ingestion Pipeline
========================================
Phase 1 of the RAG pipeline.
Loads documents from data/, chunks them, generates embeddings using
HuggingFace all-MiniLM-L6-v2, and uploads the vectors to a Pinecone index.

Usage (Docker):
    docker compose run --rm rag python ingestion.py
"""

import os
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "rag"

def load_documents(data_path: str = "data/"):
    documents = []
    pdf_loader = DirectoryLoader(
        path=data_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True,
    )
    pdf_docs = pdf_loader.load()
    print(f"   Loaded {len(pdf_docs)} pages from PDF files.")
    documents.extend(pdf_docs)
    print(f"   Total documents loaded: {len(documents)}")
    return documents

def chunk_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   Split into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap}).")
    return chunks

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    print("   HuggingFace embedding model loaded (all-MiniLM-L6-v2, 384 dims).")
    return embedding_model

def get_sparse_encoder(corpus_texts=None):
    bm25_encoder = BM25Encoder().default()
    if corpus_texts:
        print("   Fitting BM25 Encoder on corpus...")
        bm25_encoder.fit(corpus_texts)
        bm25_encoder.dump("bm25_values.json")
    else:
        if os.path.exists("bm25_values.json"):
            bm25_encoder.load("bm25_values.json")
    return bm25_encoder

def upload_to_pinecone(chunks, embedding_model, sparse_encoder):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    old_count = stats.total_vector_count
    print(f"   Pinecone index '{PINECONE_INDEX_NAME}' - current vectors: {old_count}")

    if old_count > 0:
        print(f"  ️  Deleting {old_count} existing vectors ...")
        index.delete(delete_all=True)
        print(f"   Old vectors deleted.")

    retriever = PineconeHybridSearchRetriever(
        embeddings=embedding_model,
        sparse_encoder=sparse_encoder,
        index=index
    )
    
    # We add documents in batches since Pinecone has payload limits
    print("   Uploading dense and sparse vectors...")
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        retriever.add_texts(texts[i:i+batch_size], metadatas=metadatas[i:i+batch_size])
        print(f"   Uploaded batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

    stats = index.describe_index_stats()
    print(f"   Upload complete - total vectors now: {stats.total_vector_count}")

def run_ingestion():
    print("=" * 60)
    print("  RAG Data Ingestion Pipeline (Hybrid Search)")
    print("=" * 60)

    documents = load_documents("data/")
    if not documents:
        print("    No documents found in data/. Add PDFs and re-run.")
        return False

    chunks = chunk_documents(documents)
    
    embedding_model = get_embedding_model()
    
    corpus_texts = [chunk.page_content for chunk in chunks]
    sparse_encoder = get_sparse_encoder(corpus_texts)
    
    upload_to_pinecone(chunks, embedding_model, sparse_encoder)
    return True

if __name__ == "__main__":
    run_ingestion()
