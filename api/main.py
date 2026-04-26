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
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
hybrid_retriever = HybridRetriever(
    neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    neo4j_user=os.getenv("NEO4J_USERNAME", "neo4j"),
    neo4j_password=os.getenv("NEO4J_PASSWORD", "password123"),
    weaviate_host=os.getenv("WEAVIATE_HOST", "localhost"),
    weaviate_port=int(os.getenv("WEAVIATE_PORT", 8080))
)

# --- Schemas ---

class QueryRequest(BaseModel):
    question: str
    method: str = "hybrid" # "naive", "graph", or "hybrid"
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    method: str
    fallback_reason: str = ""

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
        # 1. Retrieve Context
        result = hybrid_retriever.retrieve(request.question, mode=request.method)
        
        # 2. Generate Answer
        contexts = [c["text"] for c in result.chunks]
        context_str = "\n\n".join(contexts) if contexts else "No relevant context found."
        
        prompt = (
            f"You are a sophisticated AI analyst. Answer the question using ONLY the context provided.\n"
            f"If the context is insufficient, explain what is missing.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {request.question}\n\n"
            f"Professional Answer:"
        )
        
        response = llm.invoke(prompt)
        
        return QueryResponse(
            answer=response.content.strip(),
            method=result.mode_used,
            fallback_reason=result.fallback_reason
        )
    except Exception as e:
        print(f"[!] API Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during retrieval.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
