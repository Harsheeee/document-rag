"""
app.py — Streamlit Chat Frontend
==================================
Phase 3 of the RAG pipeline.
Provides a conversational web interface where users can ask questions
about the ingested documents. Displays answers with source citations.

Usage:
    streamlit run app.py
"""

import streamlit as st
from rag_chain import ask_question

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Domain RAG Assistant",
    page_icon="📚",
    layout="centered",
)

st.title("📚 Domain Knowledge Assistant")
st.caption(
    "Ask questions about the uploaded documents. "
    "Answers are grounded in source material and powered by a local LLM."
)

# ── Sidebar Info ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        """
        **Tech Stack:**
        - 🔗 LangChain (orchestration)
        - 🌲 Pinecone (vector DB)
        - 🤗 HuggingFace (embeddings)
        - 🦙 Ollama / Llama 3 (local LLM)
        - 🎈 Streamlit (frontend)

        **How it works:**
        1. Your documents were parsed and embedded into Pinecone.
        2. When you ask a question, the most relevant chunks are retrieved.
        3. A local LLM reads those chunks and generates an answer.
        4. Sources are cited so you can verify every claim.
        """
    )
    st.divider()
    st.markdown("Built with ❤️ using RAG")

# ── Initialize Chat History ─────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Display Existing Chat History ────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Handle New User Input ────────────────────────────────────────────────────
if user_query := st.chat_input("Ask a question about the documents..."):

    # Show the user message immediately
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Get the RAG response
    with st.spinner("🔍 Searching documents and generating answer..."):
        response = ask_question(user_query)

    answer = response["answer"]
    sources = response["sources"]

    # ── Build the assistant response with citations ──────────────────────────
    full_response = answer + "\n\n---\n**📄 Sources:**\n"
    seen_sources = set()
    for doc in sources:
        source_file = doc.metadata.get("source", "unknown")
        source_page = doc.metadata.get("page", "?")
        source_key = f"{source_file}::{source_page}"
        if source_key not in seen_sources:
            full_response += f"- `{source_file}` — page {source_page}\n"
            seen_sources.add(source_key)

    # Show the assistant message
    with st.chat_message("assistant"):
        st.markdown(full_response)

        # Show raw retrieved chunks in an expander for transparency
        with st.expander("🔎 View retrieved document chunks"):
            for i, doc in enumerate(sources):
                st.markdown(
                    f"**Chunk {i+1}** — `{doc.metadata.get('source', '?')}` "
                    f"(page {doc.metadata.get('page', '?')})"
                )
                st.text(doc.page_content)
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": full_response})
