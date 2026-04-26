import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from dotenv import load_dotenv

# Project Imports
from ingestion.pipeline import IngestionPipeline
from retrieval.hybrid_retriever import HybridRetriever
from graph.neo4j_client import Neo4jClient
from vector_store.weaviate_client import WeaviateClient
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = FastAPI(title="DeepChain Hybrid RAG API")

# --- Monitoring ---
Instrumentator().instrument(app).expose(app)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dependency Initialization ---
from vector_store.retriever import VectorRetriever
from vector_store.embedder import GeminiEmbedder
from graph.neo4j_client import Neo4jClient
from vector_store.weaviate_client import WeaviateClient

# We use the same model across the stack
LLM_MODEL = "gemini-2.5-flash"

weaviate_client = WeaviateClient()
embedder = GeminiEmbedder()
vector_retriever = VectorRetriever(weaviate_client, embedder)
neo4j_client = Neo4jClient(
    uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    user=os.getenv("NEO4J_USERNAME", "neo4j"),
    password=os.getenv("NEO4J_PASSWORD", "password123")
)

hybrid_retriever = HybridRetriever(
    retriever=vector_retriever,
    neo4j_client=neo4j_client,
    model_name=LLM_MODEL,
    top_k=5
)

# --- Schemas ---

class QueryRequest(BaseModel):
    question: str
    method: str = "auto" # "naive", "graph", "hybrid", or "auto"
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    method: str
    fallback_reason: str = ""
    latency: float = 0.0

# --- Routes ---

@app.get("/health")
def health_check():
    status = hybrid_retriever.health_check()
    return {"status": "online", "services": status}

@app.post("/ingest")
def start_ingestion():
    """Triggers the document ingestion pipeline."""
    try:
        pipeline = IngestionPipeline()
        pipeline.run()
        return {"status": "success", "message": "Ingestion completed."}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\n[!!!] INGESTION ERROR:\n{error_details}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def run_query(request: QueryRequest):
    """Executes a RAG query using the specified method."""
    try:
        # The new HybridRetriever handles retrieval + generation internally
        result = hybrid_retriever.query(
            question=request.question,
            mode=request.method,
            top_k=request.top_k
        )
        
        return QueryResponse(
            answer=result.answer,
            method=result.mode_used,
            fallback_reason=result.fallback_reason,
            latency=result.latency
        )
    except Exception as e:
        import traceback
        print(f"[!] API Error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Server Error during retrieval.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
