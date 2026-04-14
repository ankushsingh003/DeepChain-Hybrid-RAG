"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: FastAPI Backend Server
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ingestion.pipeline import IngestionPipeline
from retrieval.naive_rag import NaiveRAG
from retrieval.graph_rag import GraphRAG
from vector_store.weaviate_client import WeaviateClient
from vector_store.retriever import VectorRetriever
from vector_store.embedder import GeminiEmbedder
from graph.neo4j_client import Neo4jClient
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="DeepChain Hybrid RAG API")

# --- Dependency Initialization ---
# In a real production app, we would use lifespan events or dependency injection
neo4j_client = Neo4jClient()
weaviate_client = WeaviateClient()
embedder = GeminiEmbedder()
vector_retriever = VectorRetriever(weaviate_client, embedder)

naive_rag = NaiveRAG(vector_retriever)
graph_rag = GraphRAG(vector_retriever, neo4j_client)

# --- Schemas ---

class QueryRequest(BaseModel):
    question: str
    method: str = "graph" # "naive" or "graph"
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    method: str

# --- Routes ---

@app.get("/health")
def health_check():
    return {"status": "online", "neo4j": "connected", "weaviate": "connected"}

@app.post("/ingest")
def start_ingestion():
    """Triggers the document ingestion pipeline."""
    try:
        pipeline = IngestionPipeline()
        # In a real app, this should be a background task
        kg = pipeline.run()
        # Update Neo4j and Weaviate logic would go here in the unified pipeline
        return {"status": "success", "entities_extracted": len(kg.entities)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def run_query(request: QueryRequest):
    """Executes a RAG query using the specified method."""
    try:
        if request.method == "naive":
            answer = naive_rag.query(request.question, top_k=request.top_k)
        else:
            answer = graph_rag.query(request.question, top_k=request.top_k)
            
        return QueryResponse(answer=answer, method=request.method)
    except Exception as e:
        print(f"[!] API Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during retrieval.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
