"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Neo4j Builder - Population Logic
"""

import re
from typing import List
from graph.neo4j_client import Neo4jClient
from ingestion.extractor import KnowledgeGraph, Entity, Relationship
from graph.schema import ENTITY_LABEL

class GraphBuilder:
    def __init__(self, client: Neo4jClient):
        self.client = client

    def build_graph(self, kg: KnowledgeGraph):
        """Upserts entities and relationships into Neo4j."""
        print(f"[*] Population Phase: Writing {len(kg.entities)} entities to Neo4j...")
        
        # 1. Upsert Entities
        for entity in kg.entities:
            # We use MERGE to avoid duplicates and update descriptions if found
            cypher = (
                f"MERGE (e:{ENTITY_LABEL} {{name: $name}}) "
                f"SET e.type = $type, e.description = $description "
                f"RETURN e"
            )
            self.client.query(cypher, {
                "name": entity.name,
                "type": entity.type,
                "description": entity.description
            })

        print(f"[*] Population Phase: Writing {len(kg.relationships)} relationships to Neo4j...")
        
        # 2. Upsert Relationships
        for rel in kg.relationships:
            # MERGE on both source and target to ensure they exist, then MERGE the relationship
            # Note: Cypher doesn't allow dynamic relationship types directly in MERGE string
            # So we use a safe string injection for the relationship type (rel.type) 
            # while ensuring it's sanitized or trusted from the LLM.
            sanitized_type = re.sub(r'[^A-Z0-9_]', '', rel.type.replace(" ", "_").upper())
            if not sanitized_type:
                sanitized_type = "RELATED_TO"
            cypher = (
                f"MATCH (s:{ENTITY_LABEL} {{name: $source}}) "
                f"MATCH (t:{ENTITY_LABEL} {{name: $target}}) "
                f"MERGE (s)-[r:{sanitized_type}]->(t) "
                f"SET r.description = $description "
                f"RETURN r"
            )
            self.client.query(cypher, {
                "source": rel.source,
                "target": rel.target,
                "description": rel.description
            })
            
        print("[+] Neo4j Population complete.")

if __name__ == "__main__":
    from ingestion.extractor import Entity, Relationship
    
    # Test Builder
    client = Neo4jClient()
    builder = GraphBuilder(client)
    
    test_kg = KnowledgeGraph(
        entities=[
            Entity(name="Novatech", type="Org", description="Test org"),
            Entity(name="Rajesh", type="Person", description="Founder")
        ],
        relationships=[
            Relationship(source="Rajesh", target="Novatech", type="FOUNDED", description="Founded in 2015")
        ]
    )
    
    builder.build_graph(test_kg)
    client.close()
