"""
ingestion/extractor.py — DeepChain Hybrid-RAG
Module: Knowledge Graph Extraction Orchestrator

FIXES:
  - Replaced deprecated 'gemini-1.5-flash-latest' with 'gemini-2.0-flash'
  - Added FALLBACK_MODELS: on 404/NOT_FOUND, switches model automatically
  - Richer prompt with entity types and UPPER_SNAKE_CASE predicates
  - GraphExtractor._extract_single() now detects model-not-found and retries
    with the next fallback before raising
  - rate_limit_delay and retry_base_delay exposed as constructor params
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import List, Dict, Any, Iterator, Optional

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


def _is_model_not_found(exc: Exception) -> bool:
    msg = str(exc).upper()
    return "404" in msg or "NOT_FOUND" in msg or "NOT FOUND" in msg


# ── Schema Definitions (Pydantic for LLM Output) ──────────────────────────────

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


# ── GraphExtractor ─────────────────────────────────────────────────────────────

class GraphExtractor:
    """
    Core extractor that processes text chunks into flat triplet lists.
    Supports automatic fallback to alternative Gemini models on 404.
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_retries: int = 3,
        rate_limit_delay: float = 1.5,
        retry_base_delay: float = 5.0,
    ):
        self._model_name    = model_name or PRIMARY_MODEL
        self.max_retries    = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.retry_base_delay = retry_base_delay
        self.parser         = PydanticOutputParser(pydantic_object=KnowledgeGraph)
        self._build_chain(self._model_name)

    # ── LLM management ────────────────────────────────────────────────────────

    def _build_chain(self, model_name: str) -> None:
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self._model_name = model_name
        self.prompt = ChatPromptTemplate.from_template(
            "Extract entities and their relationships from the following text to build a knowledge graph.\n"
            "Focus on facts relevant to business, finance, medical, and legal domains.\n"
            "Use UPPER_SNAKE_CASE for relationship types (e.g. TREATS, CAUSES, OWNED_BY, LOCATED_IN).\n"
            "Keep entity names consistent — use the exact same name every time.\n"
            "{format_instructions}\n"
            "Text: {text}\n"
        )
        logger.info("[GraphExtractor] Using model: %s", model_name)

    def _switch_to_fallback(self, failed_model: str) -> bool:
        candidates = list(dict.fromkeys([PRIMARY_MODEL] + FALLBACK_MODELS))
        try:
            idx = candidates.index(failed_model)
        except ValueError:
            idx = -1
        for model in candidates[idx + 1:]:
            if model != failed_model:
                logger.warning("[GraphExtractor] '%s' not found — switching to '%s'", failed_model, model)
                self._build_chain(model)
                return True
        logger.error("[GraphExtractor] All fallback models exhausted.")
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
                return self.parser.parse(content)

            except Exception as exc:
                if _is_model_not_found(exc):
                    switched = self._switch_to_fallback(self._model_name)
                    if switched:
                        # Rebuild _input with new model prompt and retry
                        _input = self.prompt.format_prompt(
                            text=text,
                            format_instructions=self.parser.get_format_instructions()
                        )
                        logger.info("[GraphExtractor] Retrying with %s", self._model_name)
                        continue
                    else:
                        return KnowledgeGraph(entities=[], relationships=[])

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
        """Process a list of chunks → flat list of triplet dicts."""
        all_triplets = []
        seen_entity_names: set[str] = set()
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            text     = chunk.get("text", "").strip()
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            source   = chunk.get("source", "unknown")

            if not text:
                continue

            kg = self._extract_single(text)

            for entity in kg.entities:
                seen_entity_names.add(entity.name)

            for rel in kg.relationships:
                all_triplets.append({
                    "subject":        rel.source,
                    "predicate":      rel.type.upper().replace(" ", "_"),
                    "object":         rel.target,
                    "description":    rel.description,
                    "source_chunk_id": chunk_id,
                    "source_file":    source,
                })

            if i < total - 1:
                time.sleep(self.rate_limit_delay)

        return all_triplets

    def extract_batched(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 20,
    ) -> Iterator[List[Dict[str, Any]]]:
        """Generator version of extract()."""
        for start in range(0, len(chunks), batch_size):
            yield self.extract(chunks[start: start + batch_size])


# ── KnowledgeGraphExtractor (High-Level Orchestrator) ─────────────────────────

class KnowledgeGraphExtractor:
    """Extracts a KnowledgeGraph from raw document text."""

    def __init__(
        self,
        model_name: str | None = None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        model_name     = model_name or PRIMARY_MODEL
        self.extractor = GraphExtractor(model_name=model_name)
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_from_text(self, text: str, source_doc: str = "unknown") -> KnowledgeGraph:
        chunks   = self._chunk_text(text, source_doc)
        triplets = self.extractor.extract(chunks)
        return self._triplets_to_kg(triplets)

    def _chunk_text(self, text: str, source_doc: str) -> list[dict[str, Any]]:
        chunks, start, idx = [], 0, 0
        while start < len(text):
            end        = start + self.chunk_size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append({
                    "chunk_id": f"{source_doc}::chunk-{idx:04d}",
                    "text":     chunk_text,
                    "source":   source_doc,
                })
                idx += 1
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _triplets_to_kg(self, triplets: list[dict[str, str]]) -> KnowledgeGraph:
        kg_data       = {"entities": [], "relationships": []}
        seen_entities: set[str] = set()
        for t in triplets:
            subj, pred, obj = t["subject"], t["predicate"], t["object"]
            if subj not in seen_entities:
                kg_data["entities"].append(Entity(name=subj, type="Entity", description=""))
                seen_entities.add(subj)
            if obj not in seen_entities:
                kg_data["entities"].append(Entity(name=obj, type="Entity", description=""))
                seen_entities.add(obj)
            kg_data["relationships"].append(Relationship(
                source=subj, target=obj, type=pred,
                description=t.get("source_chunk_id", "")
            ))
        return KnowledgeGraph(**kg_data)
