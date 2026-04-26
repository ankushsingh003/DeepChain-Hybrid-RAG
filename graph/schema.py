# """
# DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
# Module: Graph Schema Constants
# """

# # Node Labels
# ENTITY_LABEL = "Entity"

# # Common Entity Types
# ORG_TYPE = "Organization"
# PERSON_TYPE = "Person"
# LOCATION_TYPE = "Location"
# DATE_TYPE = "Date"
# CONCEPT_TYPE = "Concept"

# # Relationship metadata keys
# CHUNK_ID_KEY = "chunk_id"
# SOURCE_DOC_KEY = "source_doc"








"""
graph/schema.py — DeepChain Hybrid-RAG

Fixes applied:
- CHUNK_ID_KEY changed from "chunk_id" to "source_chunk_id" to match the key
  that TripletExtractor.extract() actually writes onto each triplet dict.
  Previously the constant was defined but never used anywhere, and the name
  was out of sync with the extractor — now builder.py imports and uses it.
"""

# ── Node Labels ───────────────────────────────────────────────────────────────

ENTITY_LABEL = "Entity"

# ── Common Entity Types ───────────────────────────────────────────────────────

ORG_TYPE = "Organization"
PERSON_TYPE = "Person"
LOCATION_TYPE = "Location"
DATE_TYPE = "Date"
CONCEPT_TYPE = "Concept"

# ── Relationship / Property Metadata Keys ────────────────────────────────────
# FIX: was "chunk_id" — renamed to "source_chunk_id" to match the key written
# by TripletExtractor.extract() and consumed by GraphBuilder.build_graph().
CHUNK_ID_KEY = "source_chunk_id"

# Key stored on relationship properties to track the originating document
SOURCE_DOC_KEY = "source_doc"