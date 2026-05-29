"""
rag_chain.py — Retrieval & Generation Pipeline
=================================================
Phase 2 of the RAG pipeline.
Connects to the existing Pinecone index, retrieves relevant document chunks
for a user query, and generates an answer using a local Ollama LLM.

Uses the modern LangChain LCEL (LangChain Expression Language) API.
This module is imported by app.py (the Streamlit frontend).
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Load environment variables ───────────────────────────────────────────────
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "rag"

# Ollama base URL — works with host networking in Docker
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Initialize Embedding Model (must match ingestion)
# ══════════════════════════════════════════════════════════════════════════════

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Connect to Pinecone & Create Retriever
# ══════════════════════════════════════════════════════════════════════════════

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding_model,
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6},
)


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Initialize the Local LLM via Ollama
# ══════════════════════════════════════════════════════════════════════════════

llm = ChatOllama(
    model="llama3",
    temperature=0,
    base_url=OLLAMA_BASE_URL,
)


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Define the RAG Prompt Template
# ══════════════════════════════════════════════════════════════════════════════

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a precise document-analysis assistant. Your ONLY job is to answer questions using the provided document excerpts.

STRICT RULES:
1. Use ONLY the information in the CONTEXT below. Do NOT use your own knowledge.
2. Read every excerpt carefully before answering.
3. If multiple excerpts contain relevant information, synthesize them into a complete answer.
4. For every factual claim, cite the source like this: [Source: filename, page X].
5. If the context does not contain enough information to answer, respond EXACTLY with: "I don't have enough information in the provided documents to answer this question."
6. Be thorough — include all relevant details from the context, not just the first match you find.
7. Keep your answer well-structured. Use bullet points or numbered lists when listing multiple items."""),
    ("human", """CONTEXT (document excerpts):
{context}

---
QUESTION: {question}

Provide a thorough, well-cited answer based ONLY on the context above:""")
])


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Build the RAG Chain using LCEL
# ══════════════════════════════════════════════════════════════════════════════

def format_docs(docs):
    """Format retrieved documents into a numbered, clearly-labeled context string."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(
            f"EXCERPT {i} [Source: {source}, page {page}]:\n{doc.page_content}"
        )
    return "\n\n" + "\n\n---\n\n".join(formatted) + "\n"


# The LCEL chain: retrieve → format → prompt → LLM → parse
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def ask_question(query: str) -> dict:
    """
    Run the full RAG pipeline for a user query.

    Returns:
        dict with keys:
            - "answer": The LLM-generated answer grounded in retrieved context.
            - "sources": List of LangChain Document objects used as context.
    """
    # Get source docs separately for citation display
    source_docs = retriever.invoke(query)

    # Run the chain
    answer = rag_chain.invoke(query)

    return {"answer": answer, "sources": source_docs}


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    query = "Summarize the main topics covered in the documents."
    print(f"\n🔍 Query: {query}\n")

    response = ask_question(query)

    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(response["answer"])

    print("\n" + "=" * 60)
    print("SOURCES:")
    print("=" * 60)
    for doc in response["sources"]:
        print(f"  📄 {doc.metadata.get('source', 'unknown')} — page {doc.metadata.get('page', '?')}")
