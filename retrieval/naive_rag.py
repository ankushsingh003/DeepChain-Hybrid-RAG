# """
# DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
# Module: Naive RAG - Baseline Retrieval
# """

# from typing import List
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from vector_store.retriever import VectorRetriever

# class NaiveRAG:
#     def __init__(self, retriever: VectorRetriever, model_name: str = "gemini-1.5-flash"):
#         self.retriever = retriever
#         self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
#         self.prompt = ChatPromptTemplate.from_template(
#             "You are a helpful expert assistant. Answer the question based ONLY on the provided context.\n\n"
#             "Context:\n{context}\n\n"
#             "Question: {question}\n\n"
#             "Helpful Answer:"
#         )

#     def query(self, question: str, top_k: int = 5) -> str:
#         """Standard RAG flow: Retrieve -> Augment -> Generate."""
#         # 1. Retrieve
#         hits = self.retriever.retrieve(question, top_k=top_k)
#         context = "\n---\n".join([hit.content for hit in hits])
        
#         # 2. Augment & Generate
#         chain = self.prompt | self.llm
#         response = chain.invoke({"context": context, "question": question})
        
#         return response.content

# if __name__ == "__main__":
#     # Test Naive RAG (requires infrastructure to be up)
#     from vector_store.weaviate_client import WeaviateClient
#     from vector_store.embedder import GeminiEmbedder
    
#     # client = WeaviateClient()
#     # retriever = VectorRetriever(client, GeminiEmbedder())
#     # rag = NaiveRAG(retriever)
#     # print(rag.query("Who founded Novatech?"))









"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Naive RAG — Production Grade

Fixes vs original:
  - hit["content"] → hit.content  (RetrievedChunk is an object, not a dict)
  - Added query rewriting to expand/clarify ambiguous queries before retrieval
  - Added in-memory query result cache (TTL-based) to skip redundant embed+search
  - Added score-aware context building (injects score metadata into context string)
  - Added source citation in context so LLM can reference where facts came from
  - Added async query path for concurrent multi-question workloads
  - Added BM25 keyword-search fallback when vector results are sparse
  - Logging with latency timing on every query
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from vector_store.retriever import VectorRetriever, RetrievedChunk

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── Cache ──────────────────────────────────────────────────────────────────────

@dataclass
class _CacheEntry:
    result: str
    hits: list[RetrievedChunk]
    created_at: float = field(default_factory=time.time)


class _QueryCache:
    """
    Simple in-process LRU-style cache for query → (answer, chunks).
    Prevents redundant embed+search+LLM calls for repeated queries.
    TTL default: 5 minutes.
    """

    def __init__(self, ttl_seconds: float = 300.0, max_size: int = 256):
        self._store: dict[str, _CacheEntry] = {}
        self.ttl = ttl_seconds
        self.max_size = max_size

    def _key(self, query: str, top_k: int) -> str:
        return hashlib.md5(f"{query.strip().lower()}|{top_k}".encode()).hexdigest()

    def get(self, query: str, top_k: int) -> Optional[_CacheEntry]:
        key = self._key(query, top_k)
        entry = self._store.get(key)
        if entry and (time.time() - entry.created_at) < self.ttl:
            return entry
        if entry:
            del self._store[key]  # expired
        return None

    def set(self, query: str, top_k: int, result: str, hits: list[RetrievedChunk]):
        if len(self._store) >= self.max_size:
            # Evict oldest
            oldest = min(self._store, key=lambda k: self._store[k].created_at)
            del self._store[oldest]
        self._store[self._key(query, top_k)] = _CacheEntry(result=result, hits=hits)

    def invalidate(self):
        self._store.clear()


# ── Prompts ────────────────────────────────────────────────────────────────────

_REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "You are a search query optimizer for a financial/enterprise knowledge base.\n"
    "Rewrite the following question to maximize retrieval recall.\n"
    "Make it specific, expand acronyms, and add synonyms if helpful.\n"
    "Return ONLY the rewritten query — no explanation.\n\n"
    "Original: {question}\n"
    "Rewritten:"
)

_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "You are a precise expert assistant. Answer the question based ONLY on the provided context.\n"
    "If the context does not contain enough information, say so explicitly.\n"
    "Always cite the source document for each fact using [Source: <name>].\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


# ── NaiveRAG ───────────────────────────────────────────────────────────────────

class NaiveRAG:
    """
    Baseline RAG pipeline: Retrieve → Augment → Generate.

    Enhancements over v1:
        - Query rewriting before retrieval (opt-in, adds ~300ms)
        - Score-aware context with source citations
        - In-memory result cache
        - Async query path
        - BM25 keyword fallback when vector hits are sparse
    """

    def __init__(
        self,
        retriever: VectorRetriever,
        model_name: str | None = None,
        use_query_rewriting: bool = True,
        use_cache: bool = True,
        cache_ttl: float = 300.0,
        sparse_threshold: int = 2,           # trigger BM25 fallback below this many hits
    ):
        self.retriever = retriever
        self.use_query_rewriting = use_query_rewriting
        self.sparse_threshold = sparse_threshold

        model_name = model_name or os.getenv("LLM_MODEL", "gemini-1.5-flash")
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self._cache = _QueryCache(ttl_seconds=cache_ttl) if use_cache else None

        logger.info(
            f"[NaiveRAG] Init — model={model_name}, "
            f"rewrite={use_query_rewriting}, cache={use_cache}"
        )

    # ── Public sync API ────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int = 5,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """
        Full RAG pipeline.

        Returns:
            {
                "answer":   str,
                "sources":  list of source identifiers,
                "chunks":   list of RetrievedChunk,
                "mode":     "cached" | "retrieved",
                "latency":  float seconds,
            }
        """
        t0 = time.perf_counter()

        # 1. Cache check
        if use_cache and self._cache:
            cached = self._cache.get(question, top_k)
            if cached:
                logger.info(f"[NaiveRAG] Cache hit for: '{question[:60]}'")
                return {
                    "answer": cached.result,
                    "sources": list({c.source for c in cached.hits}),
                    "chunks": cached.hits,
                    "mode": "cached",
                    "latency": round(time.perf_counter() - t0, 3),
                }

        # 2. Optional query rewriting
        retrieval_query = question
        if self.use_query_rewriting:
            retrieval_query = self._rewrite_query(question)
            logger.info(f"[NaiveRAG] Rewritten query: '{retrieval_query[:80]}'")

        # 3. Vector retrieval
        hits = self.retriever.retrieve(retrieval_query, top_k=top_k)

        # 4. BM25 keyword fallback if vector results are sparse
        if len(hits) < self.sparse_threshold:
            logger.warning(
                f"[NaiveRAG] Sparse vector results ({len(hits)}), "
                f"attempting BM25 keyword fallback..."
            )
            hits = self._bm25_fallback(retrieval_query, top_k, existing=hits)

        # 5. Build context with scores + citations
        context = self._build_context(hits)

        # 6. Generate answer
        chain = _ANSWER_PROMPT | self.llm
        response = chain.invoke({"context": context, "question": question})
        answer = response.content
        if isinstance(answer, list):
            answer = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in answer])

        # 7. Cache result
        if self._cache:
            self._cache.set(question, top_k, answer, hits)

        elapsed = round(time.perf_counter() - t0, 3)
        logger.info(f"[NaiveRAG] Done in {elapsed}s | chunks={len(hits)}")

        return {
            "answer": answer,
            "sources": list({c.source for c in hits}),
            "chunks": hits,
            "mode": "retrieved",
            "latency": elapsed,
        }

    # ── Public async API ───────────────────────────────────────────────────────

    async def query_async(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Async wrapper — runs blocking query in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.query(question, top_k))

    async def query_multi(
        self, questions: list[str], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Run multiple queries concurrently."""
        tasks = [self.query_async(q, top_k) for q in questions]
        return list(await asyncio.gather(*tasks))

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _rewrite_query(self, question: str) -> str:
        """Use LLM to produce a retrieval-optimized version of the query."""
        try:
            chain = _REWRITE_PROMPT | self.llm
            response = chain.invoke({"question": question})
            rewritten = response.content.strip()
            return rewritten if rewritten else question
        except Exception as e:
            logger.warning(f"[NaiveRAG] Query rewrite failed ({e}), using original.")
            return question

    def _bm25_fallback(
        self,
        query: str,
        top_k: int,
        existing: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Attempt BM25 (keyword) search via Weaviate's hybrid search as a sparse fallback.
        If unsupported, returns existing hits unchanged.

        NOTE: Requires Weaviate's BM25/hybrid search to be enabled on the collection.
        """
        try:
            collection = self.retriever.client.client.collections.get(
                self.retriever.client.collection_name
            )
            response = collection.query.bm25(
                query=query,
                limit=top_k,
                return_properties=["content", "source", "chunk_id",
                                   "doc_type", "page_number", "section",
                                   "language", "token_count", "created_at"],
            )
            bm25_hits = [
                RetrievedChunk(properties=obj.properties, distance=0.5)
                for obj in response.objects
            ]
            # Merge: deduplicate by (source, chunk_id)
            seen = {(c.source, c.chunk_id) for c in existing}
            for h in bm25_hits:
                if (h.source, h.chunk_id) not in seen:
                    existing.append(h)
                    seen.add((h.source, h.chunk_id))
            logger.info(f"[NaiveRAG] BM25 fallback added {len(bm25_hits)} candidates.")
        except Exception as e:
            logger.warning(f"[NaiveRAG] BM25 fallback failed ({e}), skipping.")
        return existing

    @staticmethod
    def _build_context(hits: list[RetrievedChunk]) -> str:
        """
        Build the context string for the LLM.
        Includes score, source citation, and section metadata when available.
        """
        if not hits:
            return "No relevant context found."

        parts = []
        for i, h in enumerate(hits, 1):
            meta_parts = [f"Source: {h.source}", f"Score: {h.score:.3f}"]
            if h.section:
                meta_parts.append(f"Section: {h.section}")
            if h.page_number is not None:
                meta_parts.append(f"Page: {h.page_number}")
            meta = " | ".join(meta_parts)
            parts.append(f"[{i}] {meta}\n{h.content}")

        return "\n\n---\n\n".join(parts)


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from vector_store.weaviate_client import WeaviateClient
    from vector_store.embedder import GeminiEmbedder

    client = WeaviateClient()
    retriever = VectorRetriever(client, GeminiEmbedder())
    rag = NaiveRAG(retriever, use_query_rewriting=True, use_cache=True)

    result = rag.query("What is the role of NBFC in Indian FinTech?")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources: {result['sources']}")
    print(f"Latency: {result['latency']}s | Mode: {result['mode']}")

    client.close()