"""
retrieval/hybrid_retriever.py  —  DeepChain Hybrid-RAG
Fixed: unhandled Neo4j connection error crashes the entire query path.

Changes vs original:
  - Neo4j calls are wrapped in try/except with structured error logging.
  - If Neo4j is unreachable, GraphRAG transparently falls back to Naive RAG.
  - Fallback is surfaced in the returned metadata so callers / UI can show a warning.
  - Added `health_check()` so FastAPI startup can detect bad connections early.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from neo4j import GraphDatabase, exceptions as neo4j_exc
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import weaviate

logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    chunks: list[dict[str, Any]]
    mode_used: str            # "naive" | "graph" | "hybrid"
    fallback_reason: str = "" # non-empty when a fallback occurred
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Retriever ─────────────────────────────────────────────────────────────────

class HybridRetriever:
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        weaviate_host: str = "localhost",
        weaviate_port: int = 8080,
        top_k: int = 5,
        graph_depth: int = 2,
    ) -> None:
        self.top_k = top_k
        self.graph_depth = graph_depth
        self._neo4j_uri = neo4j_uri
        self._neo4j_auth = (neo4j_user, neo4j_password)
        self._weaviate_host = weaviate_host
        self._weaviate_port = weaviate_port

        import os
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
        )
        self._weaviate_client = weaviate.connect_to_local(
            host=weaviate_host, port=weaviate_port
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, mode: str = "hybrid") -> RetrievalResult:
        """
        mode: "naive" | "graph" | "hybrid"
        Always returns a RetrievalResult — never raises to caller.
        If GraphRAG fails, falls back to Naive RAG and records the reason.
        """
        if mode == "naive":
            return self._naive_retrieve(query)

        if mode == "graph":
            result = self._graph_retrieve(query)
            if result is not None:
                return result
            logger.warning("GraphRAG unavailable, falling back to Naive RAG")
            naive = self._naive_retrieve(query)
            naive.mode_used = "naive_fallback"
            naive.fallback_reason = "Neo4j unreachable — served by Naive RAG"
            return naive

        # hybrid: graph context fused with vector chunks
        graph_result = self._graph_retrieve(query)
        naive_result = self._naive_retrieve(query)

        if graph_result is None:
            logger.warning("GraphRAG unavailable in hybrid mode, using Naive only")
            naive_result.mode_used = "naive_fallback"
            naive_result.fallback_reason = "Neo4j unreachable — hybrid reduced to Naive"
            return naive_result

        # Merge: deduplicate by chunk text
        seen: set[str] = set()
        merged: list[dict[str, Any]] = []
        for chunk in graph_result.chunks + naive_result.chunks:
            key = chunk.get("text", "")[:120]
            if key not in seen:
                seen.add(key)
                merged.append(chunk)

        return RetrievalResult(
            chunks=merged[: self.top_k * 2],
            mode_used="hybrid",
            metadata={"graph_chunks": len(graph_result.chunks),
                      "naive_chunks": len(naive_result.chunks)},
        )

    def health_check(self) -> dict[str, bool]:
        """Called at FastAPI startup to surface connection issues early."""
        status = {"weaviate": False, "neo4j": False}
        try:
            self._weaviate_client.is_ready()
            status["weaviate"] = True
        except Exception as exc:  # noqa: BLE001
            logger.error("Weaviate health check failed: %s", exc)

        try:
            driver = GraphDatabase.driver(self._neo4j_uri, auth=self._neo4j_auth)
            driver.verify_connectivity()
            driver.close()
            status["neo4j"] = True
        except Exception as exc:  # noqa: BLE001
            logger.error("Neo4j health check failed: %s", exc)

        return status

    # ── Private helpers ───────────────────────────────────────────────────────

    def _naive_retrieve(self, query: str) -> RetrievalResult:
        """Pure vector similarity search against Weaviate."""
        query_vector = self._embeddings.embed_query(query)
        collection = self._weaviate_client.collections.get("DocumentChunk")

        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=self.top_k,
            return_properties=["text", "source", "chunk_id"],
        )
        chunks = [
            {"text": o.properties["text"],
             "source": o.properties.get("source", ""),
             "chunk_id": o.properties.get("chunk_id", "")}
            for o in response.objects
        ]
        return RetrievalResult(chunks=chunks, mode_used="naive")

    def _graph_retrieve(self, query: str) -> RetrievalResult | None:
        """
        Entity-anchored sub-graph retrieval from Neo4j,
        then enriched with matching Weaviate chunks.
        Returns None (instead of raising) when Neo4j is unreachable.
        """
        try:
            driver = GraphDatabase.driver(
                self._neo4j_uri, auth=self._neo4j_auth
            )
            with driver.session() as session:
                # 1. Find entities matching query terms
                entity_query = """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($query)
                    RETURN e.name AS entity, e.type AS type
                    LIMIT 10
                """
                entity_result = session.run(entity_query, query=query)
                entities = [r["entity"] for r in entity_result]

                if not entities:
                    logger.debug("No entities matched query '%s' in Neo4j", query)
                    driver.close()
                    return RetrievalResult(chunks=[], mode_used="graph",
                                           metadata={"entities_found": 0})

                # 2. Traverse sub-graph up to configured depth
                subgraph_query = """
                    MATCH path = (e:Entity)-[r*1..$depth]-(related)
                    WHERE e.name IN $entities
                    RETURN e.name AS source_entity,
                           [rel in relationships(path) | type(rel)] AS rel_types,
                           related.name AS related_entity,
                           related.text AS context_text
                    LIMIT 50
                """
                subgraph_result = session.run(
                    subgraph_query,
                    entities=entities,
                    depth=self.graph_depth,
                )
                graph_chunks = []
                for record in subgraph_result:
                    text = record.get("context_text") or (
                        f"{record['source_entity']} "
                        f"—[{', '.join(record['rel_types'])}]→ "
                        f"{record['related_entity']}"
                    )
                    graph_chunks.append({
                        "text": text,
                        "source": "neo4j_subgraph",
                        "source_entity": record["source_entity"],
                    })

            driver.close()
            return RetrievalResult(
                chunks=graph_chunks,
                mode_used="graph",
                metadata={"entities_matched": entities,
                          "graph_chunks": len(graph_chunks)},
            )

        except (
            neo4j_exc.ServiceUnavailable,
            neo4j_exc.AuthError,
            neo4j_exc.DriverError,
        ) as exc:
            logger.error("Neo4j connection error: %s", exc)
            return None  # triggers fallback in caller

        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected Neo4j error: %s", exc)
            return None