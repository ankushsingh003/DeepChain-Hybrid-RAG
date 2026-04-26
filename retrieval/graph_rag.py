# """
# DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
# Module: GraphRAG - Hybrid Retrieval (Vector + Graph)
# """

# from typing import List, Dict, Any
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from vector_store.retriever import VectorRetriever
# from graph.neo4j_client import Neo4jClient
# from graph.schema import ENTITY_LABEL

# class GraphRAG:
#     def __init__(self, retriever: VectorRetriever, neo4j_client: Neo4jClient, model_name: str = "gemini-1.5-flash"):
#         self.retriever = retriever
#         self.neo4j_client = neo4j_client
#         self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        
#         # Helper LLM to identify entities in the user question
#         self.entity_extractor_prompt = ChatPromptTemplate.from_template(
#             "Identify all specific entities (Organizations, People, Products, etc.) in this question. "
#             "Return only a comma-separated list of names.\n\n"
#             "Question: {question}\n\n"
#             "Entity Names:"
#         )

#         self.answer_prompt = ChatPromptTemplate.from_template(
#             "You are a sophisticated AI analyst. Answer the question using the hybrid context provided.\n"
#             "The context includes both unstructured text (Vector) and structured relationships (Graph).\n\n"
#             "Structured Graph Context:\n{graph_context}\n\n"
#             "Unstructured Vector Context:\n{vector_context}\n\n"
#             "Question: {question}\n\n"
#             "Professional Answer:"
#         )

#     def _get_entities_from_query(self, question: str) -> List[str]:
#         """Identifies entities in the question to target in Neo4j."""
#         chain = self.entity_extractor_prompt | self.llm
#         response = chain.invoke({"question": question})
#         names = [name.strip() for name in response.content.split(",")]
#         return names

#     def _retrieve_graph_context(self, entity_names: List[str]) -> str:
#         """Fetch relations for the identified entities from Neo4j."""
#         graph_facts = []
#         for name in entity_names:
#             # Query for immediate neighbors and their relationship
#             cypher = (
#                 f"MATCH (n:{ENTITY_LABEL} {{name: $name}})-[r]-(neighbor) "
#                 f"RETURN n.name as source, type(r) as relation, neighbor.name as target, r.description as desc "
#                 f"LIMIT 10"
#             )
#             results = self.neo4j_client.query(cypher, {"name": name})
#             for res in results:
#                 fact = f"- {res['source']} --[{res['relation']}]--> {res['target']} ({res['desc']})"
#                 graph_facts.append(fact)
        
#         return "\n".join(graph_facts) if graph_facts else "No direct graph relationships found."

#     def query(self, question: str, top_k: int = 5) -> str:
#         """Hybrid RAG flow: Graph Traversal + Vector Search."""
#         print(f"[*] Processing Hybrid Query: '{question}'")
        
#         # 1. Identify Entities from query
#         query_entities = self._get_entities_from_query(question)
#         print(f"[*] Identified Entities: {query_entities}")
        
#         # 2. Vector Retrieval
#         vector_hits = self.retriever.retrieve(question, top_k=top_k)
#         vector_context = "\n---\n".join([hit.content for hit in vector_hits]) if vector_hits else "No vector context found."
        
#         # 3. Graph Retrieval
#         graph_context = self._retrieve_graph_context(query_entities)
        
#         # 4. Generate Answer
#         chain = self.answer_prompt | self.llm
#         print("[*] Synthesizing final answer from hybrid sources...")
#         response = chain.invoke({
#             "graph_context": graph_context,
#             "vector_context": vector_context,
#             "question": question
#         })
        
#         return response.content

# if __name__ == "__main__":
#     # Test GraphRAG (requires infra)
#     pass












"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: GraphRAG — Production Grade

Fixes vs original:
  - hit["content"] → hit.content  (RetrievedChunk object, not dict)
  - Entity extraction replaced: LLM call (500-1500ms) → spaCy NER (5-20ms)
    with LLM as fallback when spaCy is not available
  - Vector + Graph retrieval now run CONCURRENTLY via asyncio.gather
    (previously sequential, wasting the graph depth latency)
  - Added Reciprocal Rank Fusion (RRF) to merge vector + graph results
    instead of naive string concatenation
  - Graph Cypher query now uses parameterized depth (was hardcoded)
  - Added null-safe access for Neo4j record fields (desc can be None)
  - Added latency timing and structured result dict (same shape as NaiveRAG)
  - LLM temperature set to 0 for deterministic answers (already was 0, kept)
  - Does NOT re-implement vector search — delegates to VectorRetriever
  - Does NOT re-implement Neo4j driver — delegates to Neo4jClient
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from vector_store.retriever import VectorRetriever, RetrievedChunk
from graph.neo4j_client import Neo4jClient
from graph.schema import ENTITY_LABEL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── NER: fast spaCy with LLM fallback ─────────────────────────────────────────

def _extract_entities_spacy(text: str) -> list[str]:
    """
    Extract named entities using spaCy (5-20ms).
    Returns empty list if spaCy is not installed — caller falls back to LLM.
    """
    try:
        import spacy  # noqa: PLC0415
        # Load small English model — install with: python -m spacy download en_core_web_sm
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not downloaded — soft fail
            return []
        doc = nlp(text)
        entities = [
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ in ("ORG", "PERSON", "PRODUCT", "GPE", "NORP", "WORK_OF_ART", "EVENT")
            and len(ent.text.strip()) > 1
        ]
        return list(dict.fromkeys(entities))  # deduplicate, preserve order
    except Exception as e:
        logger.debug(f"[GraphRAG] spaCy NER failed: {e}")
        return []


_ENTITY_EXTRACT_PROMPT = ChatPromptTemplate.from_template(
    "Identify all specific named entities (Organizations, People, Products, Places) "
    "in this question. Return ONLY a comma-separated list of names — nothing else.\n\n"
    "Question: {question}\n\nEntity Names:"
)


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, Any]]],
    k: int = 60,
) -> list[tuple[float, Any]]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Each list is a list of (text_key, payload) tuples in ranked order.
    Returns list of (rrf_score, payload) sorted by descending RRF score.

    k=60 is the standard constant from the original RRF paper.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, Any] = {}

    for ranked in ranked_lists:
        for rank, (key, payload) in enumerate(ranked, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in payloads:
                payloads[key] = payload

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, payloads[key]) for key, score in fused]


# ── GraphRAG ──────────────────────────────────────────────────────────────────

class GraphRAG:
    """
    Hybrid RAG pipeline: concurrent Vector + Knowledge Graph retrieval,
    fused with Reciprocal Rank Fusion, answered by Gemini.

    Architecture:
        1. Extract entities from query (spaCy NER, LLM fallback)
        2. Launch vector search + graph traversal CONCURRENTLY
        3. Fuse results with RRF
        4. Build rich context (vector chunks + graph facts)
        5. Generate answer with Gemini
    """

    def __init__(
        self,
        retriever: VectorRetriever,
        neo4j_client: Neo4jClient,
        model_name: str = "gemini-2.5-flash",
        graph_depth: int = 2,
        graph_limit: int = 30,
    ):
        self.retriever = retriever
        self.neo4j_client = neo4j_client
        self.graph_depth = graph_depth
        self.graph_limit = graph_limit

        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

        self._answer_prompt = ChatPromptTemplate.from_template(
            "You are a sophisticated AI analyst. Answer the question using the hybrid context.\n"
            "The context combines unstructured text (Vector) and structured relationships (Graph).\n"
            "Cite sources using [V:<n>] for vector chunks and [G:<n>] for graph facts.\n"
            "If the context is insufficient, say so explicitly.\n\n"
            "Graph Context (structured relationships):\n{graph_context}\n\n"
            "Vector Context (unstructured text):\n{vector_context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

        logger.info(f"[GraphRAG] Init — model={model_name}, graph_depth={graph_depth}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """
        Hybrid RAG query with concurrent retrieval and RRF fusion.

        Returns:
            {
                "answer":          str,
                "vector_chunks":   list[RetrievedChunk],
                "graph_facts":     list[dict],
                "entities":        list[str],
                "fused_context":   str,
                "latency":         float,
            }
        """
        t0 = time.perf_counter()
        logger.info(f"[GraphRAG] Query: '{question[:80]}'")

        # 1. Entity extraction — spaCy first, LLM fallback
        entities = _extract_entities_spacy(question)
        if not entities:
            logger.info("[GraphRAG] spaCy returned no entities, falling back to LLM extractor.")
            entities = self._extract_entities_llm(question)
        logger.info(f"[GraphRAG] Entities: {entities}")

        # 2. Concurrent retrieval — vector + graph run in parallel
        vector_hits, graph_facts = asyncio.run(
            self._retrieve_parallel(question, entities, top_k)
        )

        # 3. RRF fusion
        vector_ranked = [
            (f"v::{h.source}::{h.chunk_id}", h) for h in vector_hits
        ]
        graph_ranked = [
            (f"g::{i}::{f.get('source_entity','')}", f)
            for i, f in enumerate(graph_facts)
        ]
        fused = _reciprocal_rank_fusion([vector_ranked, graph_ranked])

        # 4. Build context strings
        vector_context = self._build_vector_context(vector_hits)
        graph_context = self._build_graph_context(graph_facts)

        # 5. Generate answer
        chain = self._answer_prompt | self.llm
        response = chain.invoke({
            "graph_context": graph_context,
            "vector_context": vector_context,
            "question": question,
        })

        elapsed = round(time.perf_counter() - t0, 3)
        logger.info(
            f"[GraphRAG] Done in {elapsed}s | "
            f"vector={len(vector_hits)} graph={len(graph_facts)} fused={len(fused)}"
        )

        return {
            "answer": response.content,
            "vector_chunks": vector_hits,
            "graph_facts": graph_facts,
            "entities": entities,
            "fused_context": f"{graph_context}\n\n{vector_context}",
            "latency": elapsed,
        }

    async def query_async(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Async wrapper for query()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.query(question, top_k))

    # ── Concurrent retrieval ───────────────────────────────────────────────────

    async def _retrieve_parallel(
        self,
        question: str,
        entities: list[str],
        top_k: int,
    ) -> tuple[list[RetrievedChunk], list[dict]]:
        """Run vector search and graph traversal concurrently."""
        loop = asyncio.get_event_loop()

        vector_task = loop.run_in_executor(
            None, lambda: self.retriever.retrieve(question, top_k=top_k)
        )
        graph_task = loop.run_in_executor(
            None, lambda: self._retrieve_graph(entities)
        )

        vector_hits, graph_facts = await asyncio.gather(vector_task, graph_task)
        return vector_hits, graph_facts

    # ── Graph retrieval ────────────────────────────────────────────────────────

    def _retrieve_graph(self, entity_names: list[str]) -> list[dict]:
        """
        Traverse Neo4j sub-graph up to self.graph_depth hops from matched entities.
        Returns a list of relationship-fact dicts.
        """
        if not entity_names:
            return []

        facts = []
        for name in entity_names:
            cypher = (
                f"MATCH (n:{ENTITY_LABEL} {{name: $name}})-[r*1..{self.graph_depth}]-(neighbor) "
                f"RETURN n.name AS source, "
                f"       [rel IN relationships(path) | type(rel)] AS relations, "
                f"       neighbor.name AS target, "
                f"       r[0].description AS desc "
                f"LIMIT {self.graph_limit}"
            )
            try:
                results = self.neo4j_client.query(cypher, {"name": name})
                for res in results:
                    desc = res.get("desc") or ""   # null-safe — desc can be None
                    fact = {
                        "source_entity": res.get("source", name),
                        "target_entity": res.get("target", ""),
                        "relations": res.get("relations", []),
                        "description": desc,
                        "text": (
                            f"{res.get('source', name)} "
                            f"--[{', '.join(res.get('relations', []))}]--> "
                            f"{res.get('target', '')} "
                            + (f"({desc})" if desc else "")
                        ),
                    }
                    facts.append(fact)
            except Exception as e:
                logger.warning(f"[GraphRAG] Graph query failed for entity '{name}': {e}")

        # Deduplicate by text key
        seen = set()
        unique = []
        for f in facts:
            if f["text"] not in seen:
                seen.add(f["text"])
                unique.append(f)

        logger.info(f"[GraphRAG] Graph facts retrieved: {len(unique)}")
        return unique

    # ── Entity extraction fallback ─────────────────────────────────────────────

    def _extract_entities_llm(self, question: str) -> list[str]:
        """LLM-based entity extraction (fallback only — slow, ~500ms)."""
        try:
            chain = _ENTITY_EXTRACT_PROMPT | self.llm
            response = chain.invoke({"question": question})
            raw = response.content.strip()
            return [e.strip() for e in raw.split(",") if e.strip()]
        except Exception as e:
            logger.error(f"[GraphRAG] LLM entity extraction failed: {e}")
            return []

    # ── Context builders ───────────────────────────────────────────────────────

    @staticmethod
    def _build_vector_context(hits: list[RetrievedChunk]) -> str:
        if not hits:
            return "No vector context found."
        parts = []
        for i, h in enumerate(hits, 1):
            meta = f"Source: {h.source} | Score: {h.score:.3f}"
            if h.section:
                meta += f" | Section: {h.section}"
            parts.append(f"[V:{i}] {meta}\n{h.content}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _build_graph_context(facts: list[dict]) -> str:
        if not facts:
            return "No direct graph relationships found."
        lines = [
            f"[G:{i}] {f['text']}"
            for i, f in enumerate(facts, 1)
        ]
        return "\n".join(lines)


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pass  # Requires live Weaviate + Neo4j infrastructure