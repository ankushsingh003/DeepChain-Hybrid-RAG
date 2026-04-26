"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Weaviate Client - Vector Database Operations
"""

import os
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class WeaviateClient:
    def __init__(self):
        self.url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        self.grpc_port = 50051 # Default gRPC port for v4
        self.client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            grpc_port=50051
        )
        self.collection_name = "DocumentChunk"
        self.create_schema()

    def close(self):
        self.client.close()

    def create_schema(self):
        """Creates the collection schema if it doesn't exist."""
        print(f"[*] Initializing Weaviate Schema: {self.collection_name}...")
        
        if self.client.collections.exists(self.collection_name):
            print(f"[!] Collection {self.collection_name} already exists.")
            return

        self.client.collections.create(
            name=self.collection_name,
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="chunk_id", data_type=DataType.INT),
            ],
            vectorizer_config=Configure.Vectorizer.none() # We provide embeddings manually
        )
        print(f"[+] Collection {self.collection_name} created.")

    def upsert_chunks(self, chunks: List[Dict[str, Any]], vectors: List[List[float]]):
        """Batch upserts text chunks and their pre-computed vectors."""
        print(f"[*] Vectorizing Phase: Writing {len(chunks)} chunks to Weaviate...")
        
        collection = self.client.collections.get(self.collection_name)
        
        with collection.batch.dynamic() as batch:
            for i, chunk in enumerate(chunks):
                batch.add_object(
                    properties=chunk,
                    vector=vectors[i]
                )
        print("[+] Weaviate Population complete.")

    def search(self, vector: List[float], limit: int = 5):
        """Performs a vector search for the given query vector."""
        collection = self.client.collections.get(self.collection_name)
        response = collection.query.near_vector(
            near_vector=vector,
            limit=limit,
            return_properties=["content", "source", "chunk_id"]
        )
        return response.objects

if __name__ == "__main__":
    # Test connection
    try:
        w_client = WeaviateClient()
        w_client.create_schema()
        print("[SUCCESS] Connected to Weaviate.")
        w_client.close()
    except Exception as e:
        print(f"[ERROR] Weaviate connection failed: {e}")
