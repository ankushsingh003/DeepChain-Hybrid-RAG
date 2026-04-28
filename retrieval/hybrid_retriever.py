# """
# retrieval/hybrid_retriever.py  —  DeepChain Hybrid-RAG
# Fixed: unhandled Neo4j connection error crashes the entire query path.

# Changes vs original:
#   - Neo4j calls are wrapped in try/except with structured error logging.
#   - If Neo4j is unreachable, GraphRAG transparently falls back to Naive RAG.
#   - Fallback is surfaced in the returned metadata so callers / UI can show a warning.
#   - Added `health_check()` so FastAPI startup can detect bad connections early.
# """

# from __future__ import annotations

# import logging
# from dataclasses import dataclass, field
# from typing import Any

# from neo4j import GraphDatabase, exceptions as neo4j_exc
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import weaviate

# logger = logging.getLogger(__name__)


# # ── Result container ──────────────────────────────────────────────────────────

# @dataclass
# class RetrievalResult:
#     chunks: list[dict[str, Any]]
#     mode_used: str            # "naive" | "graph" | "hybrid"
#     fallback_reason: str = "" # non-empty when a fallback occurred
#     metadata: dict[str, Any] = field(default_factory=dict)


# # ── Retriever ─────────────────────────────────────────────────────────────────

# class HybridRetriever:
#     def __init__(
#         self,
#         neo4j_uri: str,
#         neo4j_user: str,
#         neo4j_password: str,
#         weaviate_host: str = "localhost",
#         weaviate_port: int = 8080,
#         top_k: int = 5,
#         graph_depth: int = 2,
#     ) -> None:
#         self.top_k = top_k
#         self.graph_depth = graph_depth
#         self._neo4j_uri = neo4j_uri
#         self._neo4j_auth = (neo4j_user, neo4j_password)

#         from vector_store.weaviate_client import WeaviateClient
#         from vector_store.retriever import VectorRetriever
#         from vector_store.embedder import GeminiEmbedder

#         self._weaviate_client = WeaviateClient()
#         self._embedder = GeminiEmbedder()
#         self._vector_retriever = VectorRetriever(self._weaviate_client, self._embedder, default_top_k=top_k)

#     # ── Public API ────────────────────────────────────────────────────────────

#     def retrieve(self, query: str, mode: str = "hybrid") -> RetrievalResult:
#         """
#         mode: "naive" | "graph" | "hybrid"
#         Always returns a RetrievalResult — never raises to caller.
#         If GraphRAG fails, falls back to Naive RAG and records the reason.
#         """
#         if mode == "naive":
#             return self._naive_retrieve(query)

#         if mode == "graph":
#             result = self._graph_retrieve(query)
#             if result is not None:
#                 return result
#             logger.warning("GraphRAG unavailable, falling back to Naive RAG")
#             naive = self._naive_retrieve(query)
#             naive.mode_used = "naive_fallback"
#             naive.fallback_reason = "Neo4j unreachable — served by Naive RAG"
#             return naive

#         # hybrid: graph context fused with vector chunks
#         graph_result = self._graph_retrieve(query)
#         naive_result = self._naive_retrieve(query)

#         if graph_result is None:
#             logger.warning("GraphRAG unavailable in hybrid mode, using Naive only")
#             naive_result.mode_used = "naive_fallback"
#             naive_result.fallback_reason = "Neo4j unreachable — hybrid reduced to Naive"
#             return naive_result

#         # Merge: deduplicate by chunk text
#         seen: set[str] = set()
#         merged: list[dict[str, Any]] = []
#         for chunk in graph_result.chunks + naive_result.chunks:
#             key = chunk.get("text", "")[:120]
#             if key not in seen:
#                 seen.add(key)
#                 merged.append(chunk)

#         return RetrievalResult(
#             chunks=merged[: self.top_k * 2],
#             mode_used="hybrid",
#             metadata={"graph_chunks": len(graph_result.chunks),
#                       "naive_chunks": len(naive_result.chunks)},
#         )

#     def health_check(self) -> dict[str, bool]:
#         """Called at FastAPI startup to surface connection issues early."""
#         status = {"weaviate": False, "neo4j": False}
#         try:
#             # Weaviate v4 check
#             self._weaviate_client.client.is_ready()
#             status["weaviate"] = True
#         except Exception as exc:  # noqa: BLE001
#             logger.error("Weaviate health check failed: %s", exc)

#         try:
#             driver = GraphDatabase.driver(self._neo4j_uri, auth=self._neo4j_auth)
#             driver.verify_connectivity()
#             driver.close()
#             status["neo4j"] = True
#         except Exception as exc:  # noqa: BLE001
#             logger.error("Neo4j health check failed: %s", exc)

#         return status

#     # ── Private helpers ───────────────────────────────────────────────────────

#     def _naive_retrieve(self, query: str) -> RetrievalResult:
#         """Pure vector similarity search against Weaviate."""
#         try:
#             results = self._vector_retriever.retrieve(query, top_k=self.top_k)
#             chunks = [
#                 {
#                     "text": r.content,
#                     "source": r.source,
#                     "chunk_id": r.chunk_id,
#                     "score": r.score
#                 }
#                 for r in results
#             ]
#             return RetrievalResult(chunks=chunks, mode_used="naive")
#         except Exception as e:
#             logger.error(f"Vector retrieval failed: {e}")
#             return RetrievalResult(chunks=[], mode_used="naive", fallback_reason=str(e))

#     def _graph_retrieve(self, query: str) -> RetrievalResult | None:
#         """
#         Entity-anchored sub-graph retrieval from Neo4j,
#         then enriched with matching Weaviate chunks.
#         Returns None (instead of raising) when Neo4j is unreachable.
#         """
#         try:
#             driver = GraphDatabase.driver(
#                 self._neo4j_uri, auth=self._neo4j_auth
#             )
#             with driver.session() as session:
#                 # 1. Find entities matching query terms
#                 entity_query = """
#                     MATCH (e:Entity)
#                     WHERE toLower(e.name) CONTAINS toLower($query)
#                     RETURN e.name AS entity, e.type AS type
#                     LIMIT 10
#                 """
#                 entity_result = session.run(entity_query, query=query)
#                 entities = [r["entity"] for r in entity_result]

#                 if not entities:
#                     logger.debug("No entities matched query '%s' in Neo4j", query)
#                     driver.close()
#                     return RetrievalResult(chunks=[], mode_used="graph",
#                                            metadata={"entities_found": 0})

#                 # 2. Traverse sub-graph up to configured depth
#                 subgraph_query = """
#                     MATCH path = (e:Entity)-[r*1..$depth]-(related)
#                     WHERE e.name IN $entities
#                     RETURN e.name AS source_entity,
#                            [rel in relationships(path) | type(rel)] AS rel_types,
#                            related.name AS related_entity,
#                            related.text AS context_text
#                     LIMIT 50
#                 """
#                 subgraph_result = session.run(
#                     subgraph_query,
#                     entities=entities,
#                     depth=self.graph_depth,
#                 )
#                 graph_chunks = []
#                 for record in subgraph_result:
#                     text = record.get("context_text") or (
#                         f"{record['source_entity']} "
#                         f"—[{', '.join(record['rel_types'])}]→ "
#                         f"{record['related_entity']}"
#                     )
#                     graph_chunks.append({
#                         "text": text,
#                         "source": "neo4j_subgraph",
#                         "source_entity": record["source_entity"],
#                     })

#             driver.close()
#             return RetrievalResult(
#                 chunks=graph_chunks,
#                 mode_used="graph",
#                 metadata={"entities_matched": entities,
#                           "graph_chunks": len(graph_chunks)},
#             )

#         except (
#             neo4j_exc.ServiceUnavailable,
#             neo4j_exc.AuthError,
#             neo4j_exc.DriverError,
#         ) as exc:
#             logger.error("Neo4j connection error: %s", exc)
#             return None  # triggers fallback in caller

#         except Exception as exc:  # noqa: BLE001
#             logger.error("Unexpected Neo4j error: %s", exc)
#             return None










"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: HybridRetriever — Production Grade (Orchestrator)

The original hybrid_retriever.py had three critical architectural flaws:

  FLAW 1 — Duplication:
    It re-implemented vector search (raw Weaviate calls) and graph search
    (raw Neo4j driver) that already existed in naive_rag.py and graph_rag.py.
    Any bug fix or improvement had to be made in 3 places.

  FLAW 2 — Wrong property name:
    It queried Weaviate for property "text" but the schema stores "content".
    This caused silent empty results on every naive retrieval path.

  FLAW 3 — No LLM generation:
    hybrid_retriever.py only returned chunks (a retriever), while naive_rag.py
    and graph_rag.py returned answers (full RAG pipelines). The interface was
    inconsistent — callers had to know which one returns what.

This rewrite fixes all three by making HybridRetriever a pure ORCHESTRATOR:
  - It delegates to NaiveRAG and GraphRAG (no duplication)
  - It adds cross-encoder RERANKING on top of both pipelines
  - It selects the retrieval mode dynamically based on query characteristics
  - It exposes a unified .query() interface identical to NaiveRAG and GraphRAG
  - It retains the fallback logic when Neo4j is unavailable
  - It adds health_check() for startup validation

New additions for speed + accuracy:
  - Cross-encoder reranking (sentence-transformers) on merged chunks
  - Query classification: keyword-heavy → naive, entity-heavy → graph, mixed → hybrid
  - Async parallel execution of naive + graph pipelines
  - Unified RetrievalResult with full metadata
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from vector_store.retriever import VectorRetriever, RetrievedChunk
from graph.neo4j_client import Neo4jClient
from retrieval.naive_rag import NaiveRAG
from retrieval.graph_rag import GraphRAG, _reciprocal_rank_fusion

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RetrievalMode = Literal["naive", "graph", "hybrid", "auto"]


# ── Result container ───────────────────────────────────────────────────────────

@dataclass
class HybridResult:
    """Unified result from any retrieval mode."""
    answer: str
    chunks: list[RetrievedChunk]
    graph_facts: list[dict]
    mode_used: str
    fallback_reason: str = ""
    latency: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Cross-encoder reranker ─────────────────────────────────────────────────────

class _CrossEncoderReranker:
    """
    Optional cross-encoder reranker using sentence-transformers.
    Falls back to score-based ordering if not installed.

    Install: pip install sentence-transformers
    """

    _model = None  # class-level lazy singleton

    @classmethod
    def _load(cls):
        if cls._model is None:
            try:
                from sentence_transformers import CrossEncoder  # noqa: PLC0415
                cls._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                logger.info("[Reranker] Cross-encoder loaded: ms-marco-MiniLM-L-6-v2")
            except ImportError:
                logger.warning(
                    "[Reranker] sentence-transformers not installed. "
                    "Falling back to score-based ranking. "
                    "Install with: pip install sentence-transformers"
                )
                cls._model = "unavailable"
        return cls._model

    @classmethod
    def rerank(
        cls,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Rerank chunks using cross-encoder scores.
        Returns chunks sorted by cross-encoder relevance descending.
        If cross-encoder unavailable, returns chunks sorted by existing .score.
        """
        if not chunks:
            return chunks

        model = cls._load()
        if model == "unavailable":
            return sorted(chunks, key=lambda c: c.score, reverse=True)[:top_n]

        pairs = [(query, c.content) for c in chunks]
        scores = model.predict(pairs)

        reranked = sorted(
            zip(scores, chunks), key=lambda x: x[0], reverse=True
        )
        result = [chunk for _, chunk in reranked]
        return result[:top_n] if top_n else result


# ── Query classifier ───────────────────────────────────────────────────────────

def _classify_query(question: str) -> RetrievalMode:
    """
    Fast heuristic to pick the best retrieval mode:
      - Questions with relationship words → graph-heavy
      - Short keyword-like questions → naive
      - Everything else → hybrid

    This avoids the overhead of running both pipelines when one clearly dominates.
    """
    q = question.lower()

    graph_signals = [
        "relationship", "related to", "connected", "between", "how does",
        "who is", "founded by", "acquired", "subsidiary", "partner",
        "link", "association", "belong to", "works with",
    ]
    keyword_signals = [
        "what is", "define", "meaning of", "explain", "list",
        "how many", "when was", "what are",
    ]

    graph_hits = sum(1 for s in graph_signals if s in q)
    keyword_hits = sum(1 for s in keyword_signals if s in q)

    if graph_hits >= 2 and graph_hits > keyword_hits:
        return "graph"
    if keyword_hits >= 2 and keyword_hits > graph_hits:
        return "naive"
    return "hybrid"


# ── HybridRetriever ────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Orchestrator that combines NaiveRAG and GraphRAG into a unified interface.

    Modes:
        "naive"  — pure vector retrieval + generation (fast, no graph)
        "graph"  — entity-anchored graph traversal + vector (relationship queries)
        "hybrid" — both pipelines run concurrently, results fused with RRF + reranked
        "auto"   — automatically selects mode based on query characteristics

    Features:
        - No code duplication (delegates entirely to NaiveRAG and GraphRAG)
        - Cross-encoder reranking on final merged context
        - Concurrent execution of both pipelines in hybrid mode
        - Graceful fallback from graph → naive when Neo4j is unavailable
        - health_check() for startup validation
        - Unified HybridResult output regardless of mode
    """

    def __init__(
        self,
        retriever: VectorRetriever,
        neo4j_client: Neo4jClient,
        model_name: str | None = None,
        top_k: int = 5,
        graph_depth: int = 2,
        use_reranking: bool = True,
        use_query_rewriting: bool = True,
        use_cache: bool = True,
    ):
        self.top_k = top_k
        self.use_reranking = use_reranking

        model_name = model_name or os.getenv("LLM_MODEL", "gemini-2.0-flash")

        # Build component pipelines — no duplication
        self.naive_rag = NaiveRAG(
            retriever=retriever,
            model_name=model_name,
            use_query_rewriting=use_query_rewriting,
            use_cache=use_cache,
        )
        self.graph_rag = GraphRAG(
            retriever=retriever,
            neo4j_client=neo4j_client,
            model_name=model_name,
            graph_depth=graph_depth,
        )
        self._neo4j_client = neo4j_client
        self._vector_retriever = retriever

        logger.info(
            f"[HybridRetriever] Init — model={model_name}, top_k={top_k}, "
            f"rerank={use_reranking}, rewrite={use_query_rewriting}, cache={use_cache}"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    async def query(
        self,
        question: str,
        mode: RetrievalMode = "auto",
        top_k: int | None = None,
    ) -> HybridResult:
        """
        Run the full hybrid RAG pipeline.

        Args:
            question: Natural language query.
            mode:     "naive" | "graph" | "hybrid" | "auto"
            top_k:    Override default top_k for this query.

        Returns:
            HybridResult with answer, chunks, graph_facts, and metadata.
        """
        t0 = time.perf_counter()
        top_k = top_k or self.top_k

        # Auto-classify if requested
        effective_mode = _classify_query(question) if mode == "auto" else mode
        logger.info(f"[HybridRetriever] mode={effective_mode} | query='{question[:60]}'")

        if effective_mode == "naive":
            return await self._run_naive(question, top_k, t0)

        if effective_mode == "graph":
            result = await self._run_graph(question, top_k, t0)
            if result is not None:
                return result
            # Fallback
            logger.warning("[HybridRetriever] Graph unavailable, falling back to naive.")
            r = await self._run_naive(question, top_k, t0)
            r.mode_used = "naive_fallback"
            r.fallback_reason = "Neo4j unavailable — served by Naive RAG"
            return r

        # Hybrid: run both concurrently
        return await self._run_hybrid(question, top_k, t0)

    async def query_async(
        self,
        question: str,
        mode: RetrievalMode = "auto",
        top_k: int | None = None,
    ) -> HybridResult:
        """Async wrapper around query()."""
        return await self.query(question, mode, top_k)

    def health_check(self) -> dict[str, bool]:
        """Check connectivity of all dependent services."""
        status = {"weaviate": False, "neo4j": False}

        # Weaviate
        try:
            self._vector_retriever.client.client.is_ready()
            status["weaviate"] = True
        except Exception as e:
            logger.error(f"[HybridRetriever] Weaviate health check failed: {e}")

        # Neo4j
        try:
            from neo4j import GraphDatabase  # noqa: PLC0415
            from neo4j import exceptions as neo4j_exc  # noqa: PLC0415
            uri = self._neo4j_client.uri
            auth = (self._neo4j_client.user, self._neo4j_client.password)
            driver = GraphDatabase.driver(uri, auth=auth)
            driver.verify_connectivity()
            driver.close()
            status["neo4j"] = True
        except Exception as e:
            logger.error(f"[HybridRetriever] Neo4j health check failed: {e}")

        logger.info(f"[HybridRetriever] Health: {status}")
        return status

    # ── Mode runners ───────────────────────────────────────────────────────────

    async def _run_naive(self, question: str, top_k: int, t0: float) -> HybridResult:
        raw = self.naive_rag.query(question, top_k=top_k)
        chunks = raw["chunks"]
        if self.use_reranking:
            chunks = _CrossEncoderReranker.rerank(question, chunks, top_n=top_k)
        return HybridResult(
            answer=raw["answer"],
            chunks=chunks,
            graph_facts=[],
            mode_used="naive",
            latency=round(time.perf_counter() - t0, 3),
            metadata={"sources": raw["sources"]},
        )

    async def _run_graph(self, question: str, top_k: int, t0: float) -> HybridResult | None:
        try:
            raw = await self.graph_rag.query(question, top_k=top_k)
            chunks = raw["vector_chunks"]
            if self.use_reranking:
                chunks = _CrossEncoderReranker.rerank(question, chunks, top_n=top_k)
            return HybridResult(
                answer=raw["answer"],
                chunks=chunks,
                graph_facts=raw["graph_facts"],
                mode_used="graph",
                latency=round(time.perf_counter() - t0, 3),
                metadata={"entities": raw["entities"]},
            )
        except Exception as e:
            logger.error(f"[HybridRetriever] GraphRAG failed: {e}")
            return None

    async def _run_hybrid(self, question: str, top_k: int, t0: float) -> HybridResult:
        """
        Run naive + graph concurrently, fuse with RRF, rerank, generate answer.
        Falls back gracefully if graph is unavailable.
        """
        # Run both pipelines concurrently in thread pool
        async def _parallel():
            loop = asyncio.get_event_loop()
            naive_task = loop.run_in_executor(
                None, lambda: self.naive_rag.query(question, top_k=top_k)
            )
            graph_task = self._safe_graph_query(question, top_k)
            return await asyncio.gather(naive_task, graph_task)

        naive_raw, graph_raw = await _parallel()

        # Merge chunks
        naive_chunks: list[RetrievedChunk] = naive_raw["chunks"]
        graph_chunks: list[RetrievedChunk] = graph_raw["vector_chunks"] if graph_raw else []
        graph_facts: list[dict] = graph_raw["graph_facts"] if graph_raw else []

        fallback_reason = ""
        if graph_raw is None:
            fallback_reason = "Neo4j unavailable — hybrid reduced to naive"
            logger.warning(f"[HybridRetriever] {fallback_reason}")

        # RRF fusion on vector chunks from both pipelines
        naive_ranked = [(f"v::{c.source}::{c.chunk_id}", c) for c in naive_chunks]
        graph_vec_ranked = [(f"g::{c.source}::{c.chunk_id}", c) for c in graph_chunks]
        fused = _reciprocal_rank_fusion([naive_ranked, graph_vec_ranked])
        merged_chunks = [payload for _, payload in fused]

        # Cross-encoder rerank on merged pool
        if self.use_reranking:
            merged_chunks = _CrossEncoderReranker.rerank(
                question, merged_chunks, top_n=top_k * 2
            )

        # Use the naive answer if graph unavailable, else use graph answer
        # (graph answer incorporates both graph context and vector context)
        answer = graph_raw["answer"] if graph_raw else naive_raw["answer"]

        return HybridResult(
            answer=answer,
            chunks=merged_chunks[:top_k],
            graph_facts=graph_facts,
            mode_used="hybrid" if graph_raw else "naive_fallback",
            fallback_reason=fallback_reason,
            latency=round(time.perf_counter() - t0, 3),
            metadata={
                "naive_chunks": len(naive_chunks),
                "graph_vec_chunks": len(graph_chunks),
                "graph_facts": len(graph_facts),
                "fused_total": len(merged_chunks),
            },
        )

    async def _safe_graph_query(self, question: str, top_k: int) -> dict | None:
        """Wrapper around graph_rag.query() that returns None on any failure."""
        try:
            return await self.graph_rag.query(question, top_k=top_k)
        except Exception as e:
            logger.error(f"[HybridRetriever] Graph pipeline error: {e}")
            return None


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pass  # Requires live Weaviate + Neo4j infrastructure