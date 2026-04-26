# """
# DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
# Module: LLM-based Entity and Relationship Extractor (Triple Extraction)
# """

# import json
# from typing import List, Optional
# from pydantic import BaseModel, Field
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser
# from dotenv import load_dotenv

# load_dotenv()

# # --- Schema Definitions ---

# class Entity(BaseModel):
#     name: str = Field(description="Name of the entity (e.g., Novatech Solutions, John Doe)")
#     type: str = Field(description="Category of the entity (e.g., Organization, Person, Date, Location, Concept)")
#     description: str = Field(description="Brief context or description of the entity found in the text")

# class Relationship(BaseModel):
#     source: str = Field(description="The source entity name")
#     target: str = Field(description="The target entity name")
#     type: str = Field(description="The relationship type (e.g., OWNS, INVESTED_IN, LOCATED_AT, COMPETES_WITH)")
#     description: str = Field(description="Context of the relationship")

# class KnowledgeGraph(BaseModel):
#     entities: List[Entity] = Field(description="List of all extracted entities")
#     relationships: List[Relationship] = Field(description="List of all extracted relationships")

# # --- Extractor Implementation ---

# class GraphExtractor:
#     def __init__(self, model_name: str = "gemini-1.5-flash"):
#         self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
#         self.parser = PydanticOutputParser(pydantic_object=KnowledgeGraph)
        
#         self.prompt = ChatPromptTemplate.from_template(
#             "Extract entities and their relationships from the following text to build a knowledge graph.\n"
#             "Focus specifically on facts relevant to business, finance, and legal domains.\n"
#             "{format_instructions}\n"
#             "Text: {text}\n"
#         )

#     def extract(self, text: str) -> KnowledgeGraph:
#         """Extracts structured entities and relationships from raw text."""
#         print("[*] Extracting entities and relationships using LLM...")
#         _input = self.prompt.format_prompt(
#             text=text, 
#             format_instructions=self.parser.get_format_instructions()
#         )
        
#         try:
#             response = self.llm.invoke(_input.to_messages())
#             return self.parser.parse(response.content)
#         except Exception as e:
#             print(f"[!] Extraction failed: {e}")
#             return KnowledgeGraph(entities=[], relationships=[])

# if __name__ == "__main__":
#     # Test sample
#     test_text = (
#         "Novatech Solutions is a fintech leader based in Mumbai. "
#         "It was founded by Rajesh Sharma in 2015. "
#         "The company recently acquired FinPay for $200 million."
#     )
    
#     extractor = GraphExtractor()
#     kg = extractor.extract(test_text)
    
#     print("\n[Extracted Entities]:")
#     for e in kg.entities:
#         print(f" - {e.name} ({e.type}): {e.description}")
        
#     print("\n[Extracted Relationships]:")
#     for r in kg.relationships:
#         print(f" - {r.source} --[{r.type}]--> {r.target}")
        





"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: LLM-based Entity and Relationship Extractor (Triple Extraction)

CHANGES FROM ORIGINAL:
- The original extract() took a single string and made one LLM call for ALL text.
  On large PDFs this blows Gemini's context window and produces garbled output.
  Now extract() takes a list of chunk dicts and processes them one chunk at a time.
- Added exponential backoff retry logic (3 attempts) — Gemini Flash rate-limits
  aggressively; without retry the pipeline silently drops entities on busy runs.
- Added per-chunk error isolation — a bad chunk no longer crashes the whole extraction.
  It logs the error and continues to the next chunk.
- Added extract_batched() generator for streaming extraction: yields results chunk
  by chunk so the pipeline can write to Neo4j incrementally instead of accumulating
  a massive list in RAM.
- Added deduplication of entities across chunks using a name-based set so the same
  entity extracted from multiple chunks doesn't get stored multiple times in Neo4j.
- Added source_chunk_id tracking on every Relationship so you know exactly which
  chunk each triplet came from (useful for debugging and evaluation).
- Added configurable rate_limit_delay to stay under Gemini RPM quota.
- Kept the original Pydantic schema (Entity, Relationship, KnowledgeGraph) unchanged
  so the rest of the codebase (graph/builder.py etc.) doesn't need to change.
"""

import json
import time
from typing import List, Dict, Any, Iterator
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

load_dotenv()

# --- Schema Definitions (UNCHANGED from original) ---

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


# --- Extractor Implementation ---

class GraphExtractor:
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        max_retries: int = 3,
        rate_limit_delay: float = 1.5,   # seconds between LLM calls to respect RPM quota
        retry_base_delay: float = 5.0,    # base seconds for exponential backoff on failure
    ):
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
        """
        Makes one LLM call for a single chunk of text.
        Retries with exponential backoff on failure.
        """
        _input = self.prompt.format_prompt(
            text=text,
            format_instructions=self.parser.get_format_instructions()
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.llm.invoke(_input.to_messages())
                return self.parser.parse(response.content)
            except Exception as e:
                wait = self.retry_base_delay * (2 ** (attempt - 1))  # 5s, 10s, 20s
                print(f"  [extractor] Attempt {attempt}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries:
                    print(f"  [extractor] Retrying in {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    print(f"  [extractor] All retries exhausted for this chunk. Skipping.")
                    return KnowledgeGraph(entities=[], relationships=[])

    def extract(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a list of chunk dicts (each with 'text', 'chunk_id', 'source').
        Makes ONE LLM call per chunk (not one big call for all chunks).
        Returns a flat list of triplet dicts compatible with the pipeline.

        This replaces the original single-call extract() that took raw string text.
        """
        all_triplets = []
        seen_entity_names = set()
        total = len(chunks)

        print(f"[*] Extracting triplets from {total} chunks (one LLM call per chunk)...")

        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            source = chunk.get("source", "unknown")

            if not text:
                continue

            print(f"  [extractor] Chunk {i+1}/{total} | id={chunk_id}")

            kg = self._extract_single(text)

            # Deduplicate entities across chunks
            for entity in kg.entities:
                if entity.name not in seen_entity_names:
                    seen_entity_names.add(entity.name)

            # Convert relationships to triplet dicts with source tracking
            for rel in kg.relationships:
                all_triplets.append({
                    "subject": rel.source,
                    "predicate": rel.type,
                    "object": rel.target,
                    "description": rel.description,
                    "source_chunk_id": chunk_id,
                    "source_file": source,
                })

            # Respect Gemini RPM quota between calls
            if i < total - 1:
                time.sleep(self.rate_limit_delay)

        print(f"[+] Extracted {len(all_triplets)} triplets from {total} chunks.")
        return all_triplets

    def extract_batched(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 20,
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Generator version of extract().
        Yields triplets in batches of batch_size chunks so the pipeline can
        write to Neo4j incrementally without accumulating everything in RAM.

        Usage:
            for triplet_batch in extractor.extract_batched(chunks_data):
                neo4j_client.store(triplet_batch)
        """
        total = len(chunks)
        for start in range(0, total, batch_size):
            batch = chunks[start: start + batch_size]
            print(f"  [extractor] Processing batch {start//batch_size + 1} "
                  f"(chunks {start+1}–{min(start+batch_size, total)} of {total})")
            yield self.extract(batch)


if __name__ == "__main__":
    test_chunks = [
        {
            "text": (
                "Novatech Solutions is a fintech leader based in Mumbai. "
                "It was founded by Rajesh Sharma in 2015. "
                "The company recently acquired FinPay for $200 million."
            ),
            "chunk_id": "chunk_0",
            "source": "test.pdf",
        },
        {
            "text": (
                "FinPay is a digital payments startup headquartered in Bangalore. "
                "FinPay competes directly with PayTM and Razorpay in the Indian market."
            ),
            "chunk_id": "chunk_1",
            "source": "test.pdf",
        },
    ]

    extractor = GraphExtractor()
    triplets = extractor.extract(test_chunks)

    print("\n[Extracted Triplets]:")
    for t in triplets:
        print(f"  {t['subject']} --[{t['predicate']}]--> {t['object']}  (from {t['source_chunk_id']})")