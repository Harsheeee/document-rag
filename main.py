from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from rag_chain import ask_question
from ingestion import run_ingestion
import logging
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="Advanced RAG API")
logging.basicConfig(level=logging.INFO)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    chat_history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Convert ChatMessage list to standard langchain format
        history = []
        for msg in request.chat_history:
            if msg.role == "user":
                history.append(HumanMessage(content=msg.content))
            else:
                history.append(AIMessage(content=msg.content))
                
        result = ask_question(request.query, history)
        
        # Serialize sources (LangChain Document objects)
        serialized_sources = []
        for doc in result["sources"]:
            serialized_sources.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
            
        return ChatResponse(answer=result["answer"], sources=serialized_sources)
    except Exception as e:
        logging.error(f"Error during chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def background_ingestion_task():
    logging.info("Starting background ingestion...")
    try:
        success = run_ingestion()
        if success:
            logging.info("Background ingestion completed successfully.")
        else:
            logging.error("Background ingestion failed or found no documents.")
    except Exception as e:
        logging.error(f"Error during background ingestion: {e}")

@app.post("/ingest")
async def ingest_endpoint(background_tasks: BackgroundTasks):
    background_tasks.add_task(background_ingestion_task)
    return {"status": "Ingestion started in the background."}
