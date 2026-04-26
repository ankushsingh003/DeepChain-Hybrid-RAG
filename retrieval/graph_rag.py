"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: GraphRAG - Hybrid Retrieval (Vector + Graph)
"""

from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from vector_store.retriever import VectorRetriever
from graph.neo4j_client import Neo4jClient
from graph.schema import ENTITY_LABEL

class GraphRAG:
    def __init__(self, retriever: VectorRetriever, neo4j_client: Neo4jClient, model_name: str = "gemini-1.5-flash"):
        self.retriever = retriever
        self.neo4j_client = neo4j_client
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        
        # Helper LLM to identify entities in the user question
        self.entity_extractor_prompt = ChatPromptTemplate.from_template(
            "Identify all specific entities (Organizations, People, Products, etc.) in this question. "
            "Return only a comma-separated list of names.\n\n"
            "Question: {question}\n\n"
            "Entity Names:"
        )

        self.answer_prompt = ChatPromptTemplate.from_template(
            "You are a sophisticated AI analyst. Answer the question using the hybrid context provided.\n"
            "The context includes both unstructured text (Vector) and structured relationships (Graph).\n\n"
            "Structured Graph Context:\n{graph_context}\n\n"
            "Unstructured Vector Context:\n{vector_context}\n\n"
            "Question: {question}\n\n"
            "Professional Answer:"
        )

    def _get_entities_from_query(self, question: str) -> List[str]:
        """Identifies entities in the question to target in Neo4j."""
        chain = self.entity_extractor_prompt | self.llm
        response = chain.invoke({"question": question})
        names = [name.strip() for name in response.content.split(",")]
        return names

    def _retrieve_graph_context(self, entity_names: List[str]) -> str:
        """Fetch relations for the identified entities from Neo4j."""
        graph_facts = []
        for name in entity_names:
            # Query for immediate neighbors and their relationship
            cypher = (
                f"MATCH (n:{ENTITY_LABEL} {{name: $name}})-[r]-(neighbor) "
                f"RETURN n.name as source, type(r) as relation, neighbor.name as target, r.description as desc "
                f"LIMIT 10"
            )
            results = self.neo4j_client.query(cypher, {"name": name})
            for res in results:
                fact = f"- {res['source']} --[{res['relation']}]--> {res['target']} ({res['desc']})"
                graph_facts.append(fact)
        
        return "\n".join(graph_facts) if graph_facts else "No direct graph relationships found."

    def query(self, question: str, top_k: int = 5) -> str:
        """Hybrid RAG flow: Graph Traversal + Vector Search."""
        print(f"[*] Processing Hybrid Query: '{question}'")
        
        # 1. Identify Entities from query
        query_entities = self._get_entities_from_query(question)
        print(f"[*] Identified Entities: {query_entities}")
        
        # 2. Vector Retrieval
        vector_hits = self.retriever.retrieve(question, top_k=top_k)
        vector_context = "\n---\n".join([hit["content"] for hit in vector_hits]) if vector_hits else "No vector context found."
        
        # 3. Graph Retrieval
        graph_context = self._retrieve_graph_context(query_entities)
        
        # 4. Generate Answer
        chain = self.answer_prompt | self.llm
        print("[*] Synthesizing final answer from hybrid sources...")
        response = chain.invoke({
            "graph_context": graph_context,
            "vector_context": vector_context,
            "question": question
        })
        
        return response.content

if __name__ == "__main__":
    # Test GraphRAG (requires infra)
    pass
