"""
app.py - Streamlit Chat Frontend (API Client)
==================================
Phase 3 of the advanced RAG pipeline.
Provides a conversational web interface where users can ask questions
about the ingested documents. Displays answers with source citations.
Communicates with the FastAPI backend.
"""

import streamlit as st
import requests

# FastAPI Backend URL
API_BASE_URL = "http://localhost:8000"

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Domain RAG Assistant",
    page_icon="",
    layout="centered",
)

st.title(" Domain Knowledge Assistant")
st.caption(
    "Ask questions about the uploaded PDF documents. "
    "Answers are grounded in source material, powered by Hybrid Search, "
    "Cross-Encoder Re-ranking, and a local LLM."
)

# ── Sidebar Info ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header(" About")
    st.markdown(
        """
        **Advanced Features:**
        -  **Conversational Memory**: Remembers past turns.
        -  **Hybrid Search**: Combines BM25 & dense vectors.
        -  **Re-ranking**: Uses a Cross-Encoder for precision.
        -  **FastAPI Backend**: Decoupled asynchronous architecture.
        """
    )
    
    st.divider()
    st.subheader("Document Management")
    if st.button(" Ingest Documents", use_container_width=True):
        try:
            response = requests.post(f"{API_BASE_URL}/ingest")
            if response.status_code == 200:
                st.success("Ingestion started in the background! You can keep chatting.")
            else:
                st.error("Failed to start ingestion.")
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")
            
    st.divider()
    st.markdown("Built with  using RAG")

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
    
    # Prepare chat history to send to backend (excluding the current query)
    chat_history = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages]
    
    # Add user query to state
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Get the RAG response
    with st.spinner(" Searching documents and generating answer..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat",
                json={"query": user_query, "chat_history": chat_history}
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data["answer"]
                sources = data["sources"]
                
                # Build the assistant response with citations
                full_response = answer + "\n\n---\n** Sources:**\n"
                seen_sources = set()
                for doc in sources:
                    metadata = doc.get("metadata", {})
                    source_file = metadata.get("source", "unknown")
                    source_page = metadata.get("page", "?")
                    source_key = f"{source_file}::{source_page}"
                    if source_key not in seen_sources:
                        full_response += f"- `{source_file}` - page {source_page}\n"
                        seen_sources.add(source_key)

                # Show the assistant message
                with st.chat_message("assistant"):
                    st.markdown(full_response)

                    # Show raw retrieved chunks in an expander for transparency
                    with st.expander(" View retrieved document chunks (Top 5 post-reranking)"):
                        for i, doc in enumerate(sources):
                            metadata = doc.get("metadata", {})
                            st.markdown(
                                f"**Chunk {i+1}** - `{metadata.get('source', '?')}` "
                                f"(page {metadata.get('page', '?')})"
                            )
                            st.text(doc.get("page_content", ""))
                            st.divider()

                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.error(f"Backend Error: {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to the backend. Is FastAPI running? Error: {e}")
