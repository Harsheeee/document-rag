# Advanced Local RAG Pipeline

A production-ready, locally hosted Retrieval Augmented Generation (RAG) pipeline designed for deep, context-aware document analysis. This project allows you to ingest PDF documents, store them in a vector database, and chat with them using a local Large Language Model (LLM) through a beautiful web interface.

This project goes beyond a "simple RAG" by implementing state-of-the-art retrieval techniques including **Hybrid Search**, **Cross-Encoder Re-ranking**, and **Conversational Memory**, all orchestrated via a decoupled FastAPI backend.

---

##  Key Features

- **Hybrid Search Retrieval**: Combines standard dense semantic embeddings (sentence-transformers) with sparse keyword embeddings (BM25) to ensure high recall for both conceptual queries and exact keyword/acronym searches.
- **Cross-Encoder Re-ranking**: Initially retrieves 15 document chunks and then uses a powerful Cross-Encoder model (`ms-marco-MiniLM-L-6-v2`) to re-score and compress them down to the 5 most relevant chunks. This drastically improves LLM precision and prevents context dilution.
- **Conversational Memory**: The system remembers your chat history. You can ask follow-up questions using pronouns (e.g., *"Can you elaborate on that?"*) and the system will contextualize your query automatically.
- **Decoupled Architecture**: 
  - **Backend**: A robust FastAPI server handling orchestration, ingestion, and LLM inference.
  - **Frontend**: A lightweight Streamlit UI acting as an API client.
- **Asynchronous Ingestion**: Document ingestion runs as a FastAPI background task, meaning you can trigger an ingestion and continue chatting without freezing the UI.
- **100% Private Inference**: Utilizes a local LLM (Ollama) and local embedding models. Your document text is never sent to external service.

---

## ️ Tech Stack

- **Orchestration**: [LangChain](https://python.langchain.com/) (LangChain Classic)
- **Backend API**: [FastAPI](https://fastapi.tiangolo.com/) & Uvicorn
- **Frontend UI**: [Streamlit](https://streamlit.io/)
- **Vector Database**: [Pinecone](https://www.pinecone.io/) (Cloud-hosted index)
- **Embeddings**: 
  - Dense: [HuggingFace](https://huggingface.co/) `sentence-transformers/all-MiniLM-L6-v2`
  - Sparse: `pinecone-text` (BM25 Encoder)
- **Re-ranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM**: [Ollama](https://ollama.com/) with the `llama3` model
- **Infrastructure**: Docker & Docker Compose

---

##  Project Structure

```text
RAG/
├── data/                    # Drop your PDF documents here
├── backend/
│   ├── main.py              # FastAPI application & endpoints (/chat, /ingest)
│   ├── ingestion.py         # Chunking, BM25 fitting, and Pinecone upload logic
│   └── rag_chain.py         # Advanced RAG logic (Hybrid Search, Re-ranking, Memory)
├── app.py                   # Streamlit web frontend
├── Dockerfile               # Multi-purpose Python Docker image
├── docker-compose.yml       # Orchestrates the `api` and `ui` services
├── requirements.txt         # Python dependencies
└── .env                     # Environment variables (Pinecone API key)
```

---

##  Setup & Installation

### 1. Prerequisites
- **Docker** and **Docker Compose** installed on your system.
- **Ollama** installed and running on your host machine.
- A **Pinecone** account and API key.

### 2. Configure Ollama
Ensure you have pulled the `llama3` model via Ollama. Open your terminal and run:
```bash
ollama run llama3
```
*Note: Keep the Ollama service running in the background.*

### 3. Configure Pinecone (CRITICAL STEP)
Because this project uses **Hybrid Search** (Dense + Sparse vectors), your Pinecone index must be configured specifically for it.
1. Log into Pinecone.
2. Create an index named exactly **`rag`**.
3. Set Dimensions to **`384`** (this matches `all-MiniLM-L6-v2`).
4. Set the Metric to **`dotproduct`** (Required for Hybrid Search; `cosine` will fail).

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your Pinecone API key:
```ini
PINECONE_API_KEY=your_pinecone_api_key_here
# OLLAMA_BASE_URL is set to http://localhost:11434 by default.
```

### 5. Add Your Documents
Place the PDF files you want to query inside the `data/` directory.

---

##  Usage Instructions

### 1. Start the System
Run the following command to build the image and start both the FastAPI backend and Streamlit frontend in the background:
```bash
docker compose up --build -d
```

> ** First-Boot Warning:** The very first time the API container starts, it must download the HuggingFace Machine Learning models and NLTK tokenizer datasets. The backend will take about 30 to 60 seconds to fully boot up the first time. You can monitor the progress by running `docker compose logs -f api`.

### 2. Launch the Web App
Once the backend is fully active, open your browser and navigate to:
**http://localhost:8501**

### 3. Ingest Documents
1. In the Streamlit sidebar, click the **" Ingest Documents"** button.
2. **Important:** Pressing this button will completely **CLEAR** the existing Pinecone index and upload a fresh batch of vectors based on whatever PDFs are currently in the `data/` folder. This ensures your database never has stale chunks.
3. The ingestion runs asynchronously in the background.

### 4. Chat!
Ask complex questions. The LLM is strictly instructed to answer based *only* on the retrieved context and will cite the exact source document and page number for every factual claim. You can ask follow-up questions to dig deeper into the material.

To stop the system, run:
```bash
docker compose down
```

---

##  How It Works Under the Hood

1. **Ingestion Phase (`ingestion.py`)**:
   - Loads PDFs using LangChain's `PyPDFLoader`.
   - Chunks text aggressively (1000 tokens, 200 overlap).
   - Fits a `BM25Encoder` on the corpus to generate keyword weights.
   - Generates Dense Vectors (HuggingFace) and Sparse Vectors (BM25).
   - Uploads both to the `dotproduct` Pinecone index using `PineconeHybridSearchRetriever`.

2. **Retrieval Phase (`rag_chain.py`)**:
   - The user query goes through a `history_aware_retriever` which rewrites the question using past chat history to make it a standalone query.
   - The standalone query is embedded (Dense + Sparse) and sent to Pinecone.
   - Pinecone returns the top 15 results (balancing exact keyword hits and semantic meaning).
   - A `CrossEncoderReranker` evaluates the query against all 15 chunks and scores them for exact relevance.
   - The top 5 highest-scoring chunks are kept.

3. **Generation Phase (`rag_chain.py` & `main.py`)**:
   - The top 5 chunks are formatted into a strict system prompt.
   - `Llama 3` generates a highly accurate, grounded answer with inline citations.
   - The FastAPI backend serializes the response and sends it to the Streamlit UI.
