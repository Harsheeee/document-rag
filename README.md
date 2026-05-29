# Local RAG Pipeline

A complete, locally-hosted Retrieval-Augmented Generation (RAG) pipeline designed for domain-specific document analysis. This project allows you to ingest documents (PDFs, TXTs), store them in a vector database, and chat with them using a local Large Language Model (LLM) through a web interface.

## Tech Stack

- **Orchestration**: [LangChain](https://python.langchain.com/) (using the modern LCEL API)
- **Vector Database**: [Pinecone](https://www.pinecone.io/) (Cloud-hosted index)
- **Embeddings**: [HuggingFace](https://huggingface.co/) `sentence-transformers/all-MiniLM-L6-v2` (Local execution)
- **LLM**: [Ollama](https://ollama.com/) with the `llama3` model (Local execution)
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Infrastructure**: Docker & Docker Compose

## Project Structure

```
RAG/
├── data/                    # Drop your PDF/TXT documents here
├── app.py                   # Streamlit web application
├── ingestion.py             # Script to chunk documents and upload to Pinecone
├── rag_chain.py             # LangChain retrieval and prompt logic
├── Dockerfile               # Python 3.12 Docker image
├── docker-compose.yml       # Orchestrates the container with host networking
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (API keys)
└── .dockerignore            # Excludes unnecessary files from the build context
```

## Setup & Installation

### 1. Prerequisites
- **Docker** and **Docker Compose** installed on your system.
- **Ollama** installed on your host machine.
- A **Pinecone** account and API key.

### 2. Configure Ollama
Ensure you have pulled the `llama3` model via Ollama. Open your terminal and run:
```bash
ollama run llama3
```
*Note: Keep the Ollama service running in the background.*

### 3. Configure Environment Variables
Create a `.env` file in the root directory (if it doesn't exist) and add your Pinecone API key. You also need to have an index named `rag` created in your Pinecone dashboard with **384 dimensions** and the **Cosine** metric.
```ini
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 4. Add Your Documents
Place the PDF or text files you want to chat with inside the `data/` directory.

---

## Usage

We use Docker to avoid Python dependency conflicts. The container is configured with `network_mode: host` so it can communicate with your local Ollama instance seamlessly.

### Phase 1: Data Ingestion
Whenever you add new documents to the `data/` folder, you need to run the ingestion pipeline. This script parses the documents, splits them into semantic chunks (1000 tokens with 200 overlap), generates embeddings, and uploads them to Pinecone.

Run the ingestion script using Docker Compose:
```bash
docker compose run --rm rag python ingestion.py
```
*Note: This script automatically clears old vectors in the Pinecone index before uploading new ones to prevent stale results.*

### Phase 2: Launch the Web App
Once your documents are ingested, start the Streamlit chat interface:
```bash
docker compose up -d
```

Open your browser and navigate to:
**http://localhost:8501**

You can now ask questions! The LLM is strictly instructed to only answer based on the provided context and will cite the source document and page number for every factual claim.

### Stopping the App
To stop the Streamlit container, run:
```bash
docker compose down
```

## Why this Architecture?

- **Privacy & Cost**: Using a local LLM (Ollama) and local embeddings (HuggingFace) means your private document text is never sent to OpenAI or Anthropic, and you incur zero inference costs.
- **Accuracy**: We use large chunk sizes (1000), retrieve 6 context chunks per query, and utilize a strict system prompt that forces the LLM to ground its answers and provide inline citations.
- **Portability**: Dockerizes the Python environment to bypass complex dependency resolutions.
