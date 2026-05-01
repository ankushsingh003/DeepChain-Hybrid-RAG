"""
ingestion/extractor.py — DeepChain Hybrid-RAG
Module: Knowledge Graph Extraction Orchestrator

FIXES v3:
  - 429 RESOURCE_EXHAUSTED: parse retryDelay from Google error, sleep exactly
    that long, then retry. After MAX_QUOTA_HITS consecutive 429s on one model,
    switch to next fallback model.
  - 404 NOT_FOUND: immediate model switch.
  - FALLBACK_MODELS: gemini-2.0-flash → gemini-1.5-flash → gemini-1.5-pro → gemini-pro
  - Richer prompt with entity types + UPPER_SNAKE_CASE predicates.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import List, Dict, Any, Iterator

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Chunking config ───────────────────────────────────────────────────────────
CHUNK_SIZE    = 2500
CHUNK_OVERLAP = 200

# ── Model config ──────────────────────────────────────────────────────────────
PRIMARY_MODEL   = os.getenv("LLM_MODEL", "gemini-2.0-flash")
FALLBACK_MODELS = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-pro",
]

DEFAULT_429_WAIT = 30
MAX_429_WAIT     = 120
MAX_QUOTA_HITS   = 2


# ── Error helpers ─────────────────────────────────────────────────────────────

def _is_model_not_found(exc: Exception) -> bool:
    msg = str(exc).upper()
    return "404" in msg or "NOT_FOUND" in msg or "NOT FOUND" in msg


def _is_quota_exhausted(exc: Exception) -> bool:
    msg = str(exc).upper()
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "QUOTA" in msg


def _parse_retry_delay(exc: Exception) -> float:
    text = str(exc)
    m = re.search(r"['\"]retryDelay['\"]\s*:\s*['\"](\d+(?:\.\d+)?)s?['\"]", text)
    if m:
        return min(float(m.group(1)) + 2, MAX_429_WAIT)
    m = re.search(r"retry\s+in\s+(\d+(?:\.\d+)?)\s*s", text, re.IGNORECASE)
    if m:
        return min(float(m.group(1)) + 2, MAX_429_WAIT)
    return DEFAULT_429_WAIT


# ── Schema Definitions ────────────────────────────────────────────────────────

class Entity(BaseModel):
    name:        str = Field(description="Name of the entity")
    type:        str = Field(description="Category (Organization, Person, Date, Location, Drug, Disease, Concept …)")
    description: str = Field(description="Brief context or description found in the text")

class Relationship(BaseModel):
    source:      str = Field(description="Source entity name")
    target:      str = Field(description="Target entity name")
    type:        str = Field(description="Relationship type in UPPER_SNAKE_CASE (e.g. TREATS, OWNED_BY, LOCATED_AT)")
    description: str = Field(description="Context of the relationship")

class KnowledgeGraph(BaseModel):
    entities:      List[Entity]       = Field(description="All extracted entities")
    relationships: List[Relationship] = Field(description="All extracted relationships")


# ── GraphExtractor ────────────────────────────────────────────────────────────

class GraphExtractor:
    """
    Core extractor that processes text chunks into flat triplet lists.
    Handles 429 quota exhaustion with Google-provided retryDelay + model fallback.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_retries: int = 4,
        rate_limit_delay: float = 2.0,
        retry_base_delay: float = 5.0,
    ):
        self._model_name      = model_name or PRIMARY_MODEL
        self.max_retries      = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.retry_base_delay = retry_base_delay
        self._quota_hits      = 0
        self.parser           = PydanticOutputParser(pydantic_object=KnowledgeGraph)
        self._build_chain(self._model_name)

    # ── LLM management ────────────────────────────────────────────────────────

    def _build_chain(self, model_name: str) -> None:
        self.llm         = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self._model_name = model_name
        self._quota_hits = 0
        self.prompt      = ChatPromptTemplate.from_template(
            "Extract entities and their relationships from the following text to build a knowledge graph.\n"
            "Focus on facts relevant to business, finance, medical, and legal domains.\n"
            "Use UPPER_SNAKE_CASE for relationship types (e.g. TREATS, CAUSES, OWNED_BY, LOCATED_IN).\n"
            "Keep entity names consistent — use the exact same name every time.\n"
            "{format_instructions}\n"
            "Text: {text}\n"
        )
        logger.info("[GraphExtractor] Active model: %s", model_name)

    def _switch_to_fallback(self, failed_model: str, reason: str) -> bool:
        candidates = list(dict.fromkeys([PRIMARY_MODEL] + FALLBACK_MODELS))
        try:
            idx = candidates.index(failed_model)
        except ValueError:
            idx = -1
        for model in candidates[idx + 1:]:
            if model != failed_model:
                logger.warning("[GraphExtractor] %s — switching '%s' → '%s'",
                               reason, failed_model, model)
                self._build_chain(model)
                return True
        logger.error("[GraphExtractor] All fallback models exhausted (%s).", reason)
        return False

    # ── Single-chunk extraction ───────────────────────────────────────────────

    def _extract_single(self, text: str) -> KnowledgeGraph:
        _input = self.prompt.format_prompt(
            text=text,
            format_instructions=self.parser.get_format_instructions()
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.llm.invoke(_input.to_messages())
                content  = response.content
                if isinstance(content, list):
                    content = "".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    )
                self._quota_hits = 0   # reset on success
                return self.parser.parse(content)

            except Exception as exc:

                # 404: switch model immediately
                if _is_model_not_found(exc):
                    switched = self._switch_to_fallback(self._model_name, "404 NOT_FOUND")
                    if switched:
                        _input = self.prompt.format_prompt(
                            text=text,
                            format_instructions=self.parser.get_format_instructions()
                        )
                        continue
                    return KnowledgeGraph(entities=[], relationships=[])

                # 429: honour Google's retryDelay, then maybe switch model
                if _is_quota_exhausted(exc):
                    self._quota_hits += 1
                    wait = _parse_retry_delay(exc)
                    logger.warning(
                        "[GraphExtractor] 429 quota hit #%d on '%s'. "
                        "Sleeping %.0fs...",
                        self._quota_hits, self._model_name, wait,
                    )
                    time.sleep(wait)

                    if self._quota_hits >= MAX_QUOTA_HITS:
                        switched = self._switch_to_fallback(
                            self._model_name,
                            f"quota exhausted after {self._quota_hits} hits"
                        )
                        if switched:
                            _input = self.prompt.format_prompt(
                                text=text,
                                format_instructions=self.parser.get_format_instructions()
                            )
                        else:
                            return KnowledgeGraph(entities=[], relationships=[])
                    continue

                # Transient error: exponential backoff
                wait = self.retry_base_delay * (2 ** (attempt - 1))
                logger.warning("  [extractor] Attempt %d/%d failed: %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(wait)
                else:
                    logger.error("  [extractor] All retries exhausted for this chunk. Skipping.")
                    return KnowledgeGraph(entities=[], relationships=[])

        return KnowledgeGraph(entities=[], relationships=[])

    # ── Batch extraction ──────────────────────────────────────────────────────

    def extract(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_triplets: list[dict] = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            text     = chunk.get("text", "").strip()
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            source   = chunk.get("source", "unknown")

            if not text:
                continue

            kg = self._extract_single(text)

            for rel in kg.relationships:
                all_triplets.append({
                    "subject":         rel.source,
                    "predicate":       rel.type.upper().replace(" ", "_"),
                    "object":          rel.target,
                    "description":     rel.description,
                    "source_chunk_id": chunk_id,
                    "source_file":     source,
                })

            # Polite inter-chunk delay to stay under RPM limits
            if i < total - 1:
                time.sleep(self.rate_limit_delay)

        return all_triplets

    def extract_batched(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 20,
    ) -> Iterator[List[Dict[str, Any]]]:
        for start in range(0, len(chunks), batch_size):
            yield self.extract(chunks[start: start + batch_size])


# ── KnowledgeGraphExtractor (High-Level Orchestrator) ────────────────────────

class KnowledgeGraphExtractor:
    def __init__(
        self,
        model_name: str | None = None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        model_name         = model_name or PRIMARY_MODEL
        self.extractor     = GraphExtractor(model_name=model_name)
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_from_text(self, text: str, source_doc: str = "unknown") -> KnowledgeGraph:
        chunks   = self._chunk_text(text, source_doc)
        triplets = self.extractor.extract(chunks)
        return self._triplets_to_kg(triplets)

    def _chunk_text(self, text: str, source_doc: str) -> list[dict]:
        chunks, start, idx = [], 0, 0
        while start < len(text):
            ct = text[start: start + self.chunk_size]
            if ct.strip():
                chunks.append({"chunk_id": f"{source_doc}::chunk-{idx:04d}",
                                "text": ct, "source": source_doc})
                idx += 1
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _triplets_to_kg(self, triplets: list[dict]) -> KnowledgeGraph:
        kg_data, seen = {"entities": [], "relationships": []}, set()
        for t in triplets:
            for name in (t["subject"], t["object"]):
                if name not in seen:
                    kg_data["entities"].append(Entity(name=name, type="Entity", description=""))
                    seen.add(name)
            kg_data["relationships"].append(Relationship(
                source=t["subject"], target=t["object"],
                type=t["predicate"], description=t.get("source_chunk_id", "")
            ))
        return KnowledgeGraph(**kg_data)
