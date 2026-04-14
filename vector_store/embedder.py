"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Gemini Embedding Wrapper
"""

import os
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class GeminiEmbedder:
    def __init__(self, model: str = "models/embedding-001"):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model)

    def embed_query(self, text: str) -> List[float]:
        """Generates embedding for a single query string."""
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of document strings."""
        return self.embeddings.embed_documents(texts)

    def get_embedding_function(self):
        """Returns the underlying LangChain embedding function."""
        return self.embeddings

if __name__ == "__main__":
    # Test embedder
    embedder = GeminiEmbedder()
    vector = embedder.embed_query("FinTech growth in India")
    print(f"[TEST] Vector length: {len(vector)}")
    print(f"[TEST] First 5 dims: {vector[:5]}")
