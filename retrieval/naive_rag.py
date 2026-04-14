"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Naive RAG - Baseline Retrieval
"""

from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from vector_store.retriever import VectorRetriever

class NaiveRAG:
    def __init__(self, retriever: VectorRetriever, model_name: str = "gemini-1.5-flash"):
        self.retriever = retriever
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self.prompt = ChatPromptTemplate.from_template(
            "You are a helpful expert assistant. Answer the question based ONLY on the provided context.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Helpful Answer:"
        )

    def query(self, question: str, top_k: int = 5) -> str:
        """Standard RAG flow: Retrieve -> Augment -> Generate."""
        # 1. Retrieve
        hits = self.retriever.retrieve(question, top_k=top_k)
        context = "\n---\n".join([hit["content"] for hit in hits])
        
        # 2. Augment & Generate
        chain = self.prompt | self.llm
        response = chain.invoke({"context": context, "question": question})
        
        return response.content

if __name__ == "__main__":
    # Test Naive RAG (requires infrastructure to be up)
    from vector_store.weaviate_client import WeaviateClient
    from vector_store.embedder import GeminiEmbedder
    
    # client = WeaviateClient()
    # retriever = VectorRetriever(client, GeminiEmbedder())
    # rag = NaiveRAG(retriever)
    # print(rag.query("Who founded Novatech?"))
