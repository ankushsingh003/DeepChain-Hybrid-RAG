# """
# DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
# Module: Neo4j Builder - Population Logic
# """

# import re
# from typing import List
# from graph.neo4j_client import Neo4jClient
# from ingestion.extractor import KnowledgeGraph, Entity, Relationship
# from graph.schema import ENTITY_LABEL

# class GraphBuilder:
#     def __init__(self, client: Neo4jClient):
#         self.client = client

#     def build_graph(self, kg: KnowledgeGraph):
#         """Upserts entities and relationships into Neo4j."""
#         print(f"[*] Population Phase: Writing {len(kg.entities)} entities to Neo4j...")
        
#         # 1. Upsert Entities
#         for entity in kg.entities:
#             # We use MERGE to avoid duplicates and update descriptions if found
#             cypher = (
#                 f"MERGE (e:{ENTITY_LABEL} {{name: $name}}) "
#                 f"SET e.type = $type, e.description = $description "
#                 f"RETURN e"
#             )
#             self.client.query(cypher, {
#                 "name": entity.name,
#                 "type": entity.type,
#                 "description": entity.description
#             })

#         print(f"[*] Population Phase: Writing {len(kg.relationships)} relationships to Neo4j...")
        
#         # 2. Upsert Relationships
#         for rel in kg.relationships:
#             # MERGE on both source and target to ensure they exist, then MERGE the relationship
#             # Note: Cypher doesn't allow dynamic relationship types directly in MERGE string
#             # So we use a safe string injection for the relationship type (rel.type) 
#             # while ensuring it's sanitized or trusted from the LLM.
#             sanitized_type = re.sub(r'[^A-Z0-9_]', '', rel.type.replace(" ", "_").upper())
#             if not sanitized_type:
#                 sanitized_type = "RELATED_TO"
#             cypher = (
#                 f"MATCH (s:{ENTITY_LABEL} {{name: $source}}) "
#                 f"MATCH (t:{ENTITY_LABEL} {{name: $target}}) "
#                 f"MERGE (s)-[r:{sanitized_type}]->(t) "
#                 f"SET r.description = $description "
#                 f"RETURN r"
#             )
#             self.client.query(cypher, {
#                 "source": rel.source,
#                 "target": rel.target,
#                 "description": rel.description
#             })
            
#         print("[+] Neo4j Population complete.")

# if __name__ == "__main__":
#     from ingestion.extractor import Entity, Relationship
    
#     # Test Builder
#     client = Neo4jClient()
#     builder = GraphBuilder(client)
    
#     test_kg = KnowledgeGraph(
#         entities=[
#             Entity(name="Novatech", type="Org", description="Test org"),
#             Entity(name="Rajesh", type="Person", description="Founder")
#         ],
#         relationships=[
#             Relationship(source="Rajesh", target="Novatech", type="FOUNDED", description="Founded in 2015")
#         ]
#     )
    
#     builder.build_graph(test_kg)
#     client.close()






"""
graph/builder.py — DeepChain Hybrid-RAG

Fixes applied:
- Removed broken import from ingestion.extractor (KnowledgeGraph / Entity /
  Relationship). Those dataclasses no longer exist in the new graph/extractor.py.
- build_graph() now accepts the flat triplet list produced by TripletExtractor.extract().
  Each triplet is a dict with keys: subject, predicate, object, source_chunk_id.
- Relationship type sanitization kept as-is (already correct).
- Schema constants (CHUNK_ID_KEY) now imported from graph.schema and used
  consistently so property names stay in sync across the codebase.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from graph.neo4j_client import Neo4jClient
from graph.schema import ENTITY_LABEL, CHUNK_ID_KEY

logger = logging.getLogger(__name__)


class GraphBuilder:

    def __init__(self, client: Neo4jClient) -> None:
        self.client = client

    # ── Public API ────────────────────────────────────────────────────────────

    def build_graph(self, triplets: list[dict[str, Any]]) -> None:
        """
        Upserts entities and relationships into Neo4j from a flat triplet list.

        Each triplet dict must contain:
            subject        (str) — source entity name
            predicate      (str) — relationship label (will be sanitized)
            object         (str) — target entity name
            source_chunk_id (str, optional) — provenance chunk ID

        This matches the output of TripletExtractor.extract() in graph/extractor.py.
        """
        logger.info("[*] Population Phase: writing %d triplets to Neo4j...", len(triplets))

        skipped = 0

        for t in triplets:
            subject = (t.get("subject") or "").strip()
            predicate = (t.get("predicate") or "").strip()
            obj = (t.get("object") or "").strip()
            chunk_id = t.get("source_chunk_id", "unknown")

            # Skip any malformed triplet that slipped through extraction
            if not (subject and predicate and obj):
                logger.debug("Skipping incomplete triplet: %s", t)
                skipped += 1
                continue

            # 1. Upsert subject node
            self.client.query(
                f"MERGE (e:{ENTITY_LABEL} {{name: $name}}) RETURN e",
                {"name": subject},
            )

            # 2. Upsert object node
            self.client.query(
                f"MERGE (e:{ENTITY_LABEL} {{name: $name}}) RETURN e",
                {"name": obj},
            )

            # 3. Sanitize predicate → valid Cypher relationship type
            # Replace spaces with underscores, strip non-alphanumeric chars, uppercase.
            sanitized_type = re.sub(
                r"[^A-Z0-9_]", "", predicate.replace(" ", "_").upper()
            )
            if not sanitized_type:
                sanitized_type = "RELATED_TO"

            # 4. Upsert the relationship, tagging it with provenance chunk ID
            self.client.query(
                f"MATCH (s:{ENTITY_LABEL} {{name: $source}}) "
                f"MATCH (t:{ENTITY_LABEL} {{name: $target}}) "
                f"MERGE (s)-[r:{sanitized_type}]->(t) "
                f"SET r.{CHUNK_ID_KEY} = $chunk_id "
                f"RETURN r",
                {
                    "source": subject,
                    "target": obj,
                    "chunk_id": chunk_id,
                },
            )

        logger.info(
            "[+] Neo4j population complete — written=%d skipped=%d",
            len(triplets) - skipped,
            skipped,
        )
        print(f"[+] Neo4j population complete — written={len(triplets) - skipped}, skipped={skipped}")


# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Directly test builder with hand-crafted triplets (no LLM needed)
    client = Neo4jClient()
    builder = GraphBuilder(client)

    test_triplets = [
        {
            "subject": "Novatech",
            "predicate": "FOUNDED_BY",
            "object": "Rajesh",
            "source_chunk_id": "test-chunk-001",
        },
        {
            "subject": "Rajesh",
            "predicate": "LEADS",
            "object": "Novatech",
            "source_chunk_id": "test-chunk-001",
        },
    ]

    builder.build_graph(test_triplets)
    client.close()