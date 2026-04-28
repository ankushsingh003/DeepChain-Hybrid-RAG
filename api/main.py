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
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

# --- Lazy Initialization Helpers ---
# We wrap these to prevent the app from crashing on start (Status 1) if DBs are offline.

_hybrid_retriever = None

def get_retriever():
    global _hybrid_retriever
    if _hybrid_retriever is None:
        try:
            w_client = WeaviateClient()
            emb = GeminiEmbedder()
            v_retriever = VectorRetriever(w_client, emb)
            n_client = Neo4jClient(
                uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                user=os.getenv("NEO4J_USERNAME", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", "password123")
            )
            _hybrid_retriever = HybridRetriever(
                retriever=v_retriever,
                neo4j_client=n_client,
                model_name=LLM_MODEL,
                top_k=5
            )
        except Exception as e:
            print(f"[!] Critical Error: Could not initialize RAG components: {e}")
            # We don't raise here, so the FastAPI app can still start and show health status
            return None
    return _hybrid_retriever

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
    retriever = get_retriever()
    if not retriever:
        return {"status": "degraded", "services": {"database": "offline"}}
    status = retriever.health_check()
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
async def run_query(request: QueryRequest):
    """Executes a RAG query using the specified method."""
    try:
        retriever = get_retriever()
        if not retriever:
            raise HTTPException(status_code=503, detail="Database services are currently unavailable.")
            
        # The new HybridRetriever handles retrieval + generation internally
        result = await retriever.query(
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
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
