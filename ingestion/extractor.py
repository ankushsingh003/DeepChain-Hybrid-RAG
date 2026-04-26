"""
ingestion/extractor.py — DeepChain Hybrid-RAG
Module: Knowledge Graph Extraction Orchestrator

Contains:
1. Pydantic models (Entity, Relationship, KnowledgeGraph) for structured graph data.
2. GraphExtractor: Low-level chunk-by-chunk extractor with batching and retries.
3. KnowledgeGraphExtractor: High-level document-level orchestrator.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Iterator, Optional

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Chunking config ───────────────────────────────────────────────────────────
CHUNK_SIZE = 2500   # characters
CHUNK_OVERLAP = 200  # characters


# ── Schema Definitions (Pydantic for LLM Output) ───────────────────────────────

class Entity(BaseModel):
    name: str = Field(description="Name of the entity (e.g., Novatech Solutions, John Doe)")
    type: str = Field(description="Category of the entity (e.g., Organization, Person, Date, Location, Concept)")
    description: str = Field(description="Brief context or description of the entity found in the text")

class Relationship(BaseModel):
    source: str = Field(description="The source entity name")
    target: str = Field(description="The target entity name")
    type: str = Field(description="The relationship type (e.g., OWNS, INVESTED_IN, LOCATED_AT, COMPETES_WITH)")
    description: str = Field(description="Context of the relationship")

class KnowledgeGraph(BaseModel):
    entities: List[Entity] = Field(description="List of all extracted entities")
    relationships: List[Relationship] = Field(description="List of all extracted relationships")


# ── Extractor Implementation (Low Level) ───────────────────────────────────────

class GraphExtractor:
    """
    Core extractor that processes text chunks into flat triplet lists.
    Used by IngestionPipeline for streaming extraction.
    """
    def __init__(
        self,
        model_name: str | None = None,
        max_retries: int = 3,
        rate_limit_delay: float = 1.5,   # seconds between LLM calls
        retry_base_delay: float = 5.0,    # base seconds for backoff
    ):
        model_name = model_name or os.getenv("LLM_MODEL", "gemini-1.5-flash")
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.retry_base_delay = retry_base_delay

        self.prompt = ChatPromptTemplate.from_template(
            "Extract entities and their relationships from the following text to build a knowledge graph.\n"
            "Focus specifically on facts relevant to business, finance, and legal domains.\n"
            "Keep entity names consistent — use the exact same name every time you see the same entity.\n"
            "{format_instructions}\n"
            "Text: {text}\n"
        )

    def _extract_single(self, text: str) -> KnowledgeGraph:
        """Makes one LLM call for a single chunk of text."""
        _input = self.prompt.format_prompt(
            text=text,
            format_instructions=self.parser.get_format_instructions()
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.llm.invoke(_input.to_messages())
                
                # Handle cases where response.content might be a list of dicts (parts) 
                # instead of a plain string in newer langchain-google-genai versions.
                content = response.content
                if isinstance(content, list):
                    content = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
                
                return self.parser.parse(content)
            except Exception as e:
                wait = self.retry_base_delay * (2 ** (attempt - 1))
                logger.warning(f"  [extractor] Attempt {attempt}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(wait)
                else:
                    logger.error("  [extractor] All retries exhausted for this chunk. Skipping.")
                    return KnowledgeGraph(entities=[], relationships=[])

    def extract(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Processes a list of chunks and returns a flat list of triplet dicts."""
        all_triplets = []
        seen_entity_names = set()
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            source = chunk.get("source", "unknown")

            if not text:
                continue

            kg = self._extract_single(text)

            for entity in kg.entities:
                if entity.name not in seen_entity_names:
                    seen_entity_names.add(entity.name)

            for rel in kg.relationships:
                all_triplets.append({
                    "subject": rel.source,
                    "predicate": rel.type,
                    "object": rel.target,
                    "description": rel.description,
                    "source_chunk_id": chunk_id,
                    "source_file": source,
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
        total = len(chunks)
        for start in range(0, total, batch_size):
            batch = chunks[start: start + batch_size]
            yield self.extract(batch)


# ── KnowledgeGraphExtractor (High Level Orchestrator) ──────────────────────────

class KnowledgeGraphExtractor:
    """
    Extracts a KnowledgeGraph from raw document text.
    Uses TripletExtractor internally (import from graph.extractor for core logic).
    """

    def __init__(
        self,
        model_name: str | None = None,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> None:
        model_name = model_name or os.getenv("LLM_MODEL", "gemini-1.5-flash")
        self.extractor = GraphExtractor(model_name=model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_from_text(self, text: str, source_doc: str = "unknown") -> KnowledgeGraph:
        """Splits text and extracts a full KnowledgeGraph."""
        chunks = self._chunk_text(text, source_doc)
        triplets = self.extractor.extract(chunks)
        return self._triplets_to_kg(triplets)

    def _chunk_text(self, text: str, source_doc: str) -> list[dict[str, Any]]:
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append({
                    "chunk_id": f"{source_doc}::chunk-{idx:04d}",
                    "text": chunk_text,
                    "source": source_doc,
                })
                idx += 1
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _triplets_to_kg(self, triplets: list[dict[str, str]]) -> KnowledgeGraph:
        kg_data = {"entities": [], "relationships": []}
        seen_entities = set()
        for t in triplets:
            subj, pred, obj = t["subject"], t["predicate"], t["object"]
            if subj not in seen_entities:
                kg_data["entities"].append(Entity(name=subj, type="Entity", description=""))
                seen_entities.add(subj)
            if obj not in seen_entities:
                kg_data["entities"].append(Entity(name=obj, type="Entity", description=""))
                seen_entities.add(obj)
            kg_data["relationships"].append(Relationship(
                source=subj, target=obj, type=pred, description=t.get("source_chunk_id", "")
            ))
        return KnowledgeGraph(**kg_data)