import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from pinecone_text.sparse import BM25Encoder

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "rag"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# 1. Embeddings & Sparse Encoder
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
)

if os.path.exists("bm25_values.json"):
    sparse_encoder = BM25Encoder().default()
    sparse_encoder.load("bm25_values.json")
else:
    sparse_encoder = BM25Encoder().default()

# 2. Pinecone Hybrid Search Retriever
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

base_retriever = PineconeHybridSearchRetriever(
    embeddings=embedding_model,
    sparse_encoder=sparse_encoder,
    index=index,
    top_k=15, # Retrieve more chunks initially for re-ranking
)

# 3. Cross-Encoder Re-ranker
cross_encoder_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
compressor = CrossEncoderReranker(model=cross_encoder_model, top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=base_retriever
)

# 4. LLM
llm = ChatOllama(
    model="llama3",
    temperature=0,
    base_url=OLLAMA_BASE_URL,
)

# 5. History-Aware Retriever
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, compression_retriever, contextualize_q_prompt
)

# 6. Question Answering Chain
qa_system_prompt = (
    "You are a precise document-analysis assistant. Your ONLY job is to answer questions using the provided document excerpts.\n\n"
    "STRICT RULES:\n"
    "1. Use ONLY the information in the CONTEXT below. Do NOT use your own knowledge.\n"
    "2. If the context does not contain enough information to answer, respond EXACTLY with: 'I don't have enough information in the provided documents to answer this question.'\n"
    "3. Be thorough - include all relevant details from the context.\n"
    "4. For every factual claim, cite the source like this: [Source: filename, page X].\n\n"
    "CONTEXT (document excerpts):\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# 7. Final RAG Chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def ask_question(query: str, chat_history: list = None) -> dict:
    if chat_history is None:
        chat_history = []
        
    response = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history
    })
    
    return {
        "answer": response["answer"],
        "sources": response["context"] # The retrieved documents
    }
