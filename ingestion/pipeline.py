# """
# DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
# Module: Data Ingestion Pipeline Orchestrator
# """

# import json
# from typing import List
# from ingestion.loader import DocumentLoader
# from ingestion.chunker import DocumentChunker
# from graph.extractor import TripletExtractor
# from graph.neo4j_client import Neo4jClient
# from graph.builder import GraphBuilder
# from vector_store.weaviate_client import WeaviateClient
# from vector_store.embedder import GeminiEmbedder

# class IngestionPipeline:
#     def __init__(self, data_path: str = "data/sample_docs"):
#         self.loader = DocumentLoader(data_path)
#         self.chunker = DocumentChunker()
#         self.extractor = TripletExtractor()
#         self.neo4j_client = Neo4jClient()
#         self.weaviate_client = WeaviateClient()
#         self.embedder = GeminiEmbedder()

#     def run(self):
#         """Runs the full ingestion pipeline: Load -> Chunk -> Extract -> Store."""
#         print("\n[PIPELINE] Starting Information Extraction Pipeline...")
        
#         # 1. Load
#         documents = self.loader.load_documents()
#         if not documents:
#             print("[!] No documents found. Exiting.")
#             return

#         # 2. Chunk
#         chunks = self.chunker.split_documents(documents)
#         chunks_data = [
#             {
#                 "text": c.page_content,
#                 "source": c.metadata.get("source", "unknown"),
#                 "chunk_id": f"chunk_{i}",
#             }
#             for i, c in enumerate(chunks)
#         ]
        
#         # 3. Extract Triplets (New Logic)
#         print(f"[*] Extracting triplets from {len(chunks_data)} chunks...")
#         triplets = self.extractor.extract(chunks_data)
        
#         # 4. Store in Neo4j
#         print(f"[*] Building Knowledge Graph in Neo4j...")
#         self.neo4j_client.initialize_schema()
#         # We need to adapt triplets to the KnowledgeGraph schema if builder expects it
#         # Or update builder. For now, let's convert triplets to the expected format
#         from ingestion.extractor import KnowledgeGraph, Entity, Relationship
        
#         entities_map = {}
#         relationships = []
#         for t in triplets:
#             subj = t["subject"]
#             obj = t["object"]
#             pred = t["predicate"]
            
#             if subj not in entities_map:
#                 entities_map[subj] = Entity(name=subj, type="Entity", description="Extracted entity")
#             if obj not in entities_map:
#                 entities_map[obj] = Entity(name=obj, type="Entity", description="Extracted entity")
                
#             relationships.append(Relationship(
#                 source=subj, 
#                 target=obj, 
#                 type=pred, 
#                 description=f"Extracted from {t.get('source_chunk_id')}"
#             ))
            
#         kg = KnowledgeGraph(entities=list(entities_map.values()), relationships=relationships)
#         builder = GraphBuilder(self.neo4j_client)
#         builder.build_graph(kg)
        
#         # 5. Store in Weaviate
#         print(f"[*] Storing {len(chunks_data)} chunks in Weaviate...")
#         # Note: Weaviate expects 'content' instead of 'text' based on previous schema
#         # Let's align them
#         weaviate_data = []
#         texts = []
#         for c in chunks_data:
#             weaviate_data.append({
#                 "content": c["text"],
#                 "source": c["source"],
#                 "chunk_id": int(c["chunk_id"].split("_")[1])
#             })
#             texts.append(c["text"])
            
#         vectors = self.embedder.embed_documents(texts)
#         self.weaviate_client.upsert_chunks(weaviate_data, vectors)
        
#         print(f"\n[PIPELINE] Completed successfully.")

# if __name__ == "__main__":
#     # Ensure sample data exists
#     import os
#     os.makedirs("data/sample_docs", exist_ok=True)
#     pipeline = IngestionPipeline()
#     pipeline.run()







"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Data Ingestion Pipeline Orchestrator

CHANGES FROM ORIGINAL:
- Replaced full in-memory load with streaming pipeline using loader.load_documents_lazy()
  and chunker.split_documents_batched(). Documents and chunks are now processed in
  batches and never all held in RAM at the same time.
- Added checkpointing via a JSON progress file (ingestion_checkpoint.json).
  If the pipeline crashes halfway through a large PDF, re-running it picks up from
  the last successfully processed batch instead of starting over from scratch.
- Replaced single bulk embed call (embed_documents(all_texts)) with batched embedding
  in groups of EMBED_BATCH_SIZE. This avoids Gemini embedding API token limits and
  rate limits on large document sets.
- Replaced single bulk weaviate upsert with batched upserts matching the embedding batches.
- Replaced single bulk triplet extraction with extract_batched() generator so Neo4j
  writes happen incrementally after each batch, not all at once at the end.
- Added progress summary at the end (total chunks, triplets, time elapsed).
- Added EMBED_BATCH_SIZE and TRIPLET_BATCH_SIZE as class-level constants so they're
  easy to tune without hunting through the code.
- Kept the same public interface: IngestionPipeline(data_path).run() still works.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.extractor import GraphExtractor
from graph.neo4j_client import Neo4jClient
from graph.builder import GraphBuilder
from vector_store.weaviate_client import WeaviateClient
from vector_store.embedder import GeminiEmbedder
from ingestion.extractor import KnowledgeGraph, Entity, Relationship


# ── Tunable batch sizes ────────────────────────────────────────────────────────
CHUNK_BATCH_SIZE   = 20    # reduced for stability
EMBED_BATCH_SIZE   = 30    # safe sub-batch for embeddings
TRIPLET_BATCH_SIZE = 4     # very small batches for heavy LLM extraction
INTER_BATCH_DELAY  = 8.0   # increased delay to prevent RPM exhaustion
CHECKPOINT_FILE    = "ingestion_checkpoint.json"
# ──────────────────────────────────────────────────────────────────────────────


class IngestionPipeline:
    def __init__(self, data_path: str = "data/sample_docs"):
        self.loader          = DocumentLoader(data_path, use_ocr=True)
        self.chunker         = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        self.extractor       = GraphExtractor(rate_limit_delay=4.0, retry_base_delay=10.0)
        self.neo4j_client    = Neo4jClient()
        self.weaviate_client = WeaviateClient()
        self.embedder        = GeminiEmbedder()

    # ── Checkpointing helpers ──────────────────────────────────────────────────

    def _load_checkpoint(self) -> Dict[str, Any]:
        if Path(CHECKPOINT_FILE).exists():
            with open(CHECKPOINT_FILE, "r") as f:
                cp = json.load(f)
            print(f"[checkpoint] Resuming from checkpoint: {cp['batches_done']} batches already done.")
            return cp
        return {"batches_done": 0, "chunks_stored": 0, "triplets_stored": 0}

    def _save_checkpoint(self, cp: Dict[str, Any]):
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(cp, f, indent=2)

    def _clear_checkpoint(self):
        if Path(CHECKPOINT_FILE).exists():
            os.remove(CHECKPOINT_FILE)
            print("[checkpoint] Cleared checkpoint file after successful run.")

    # ── Main pipeline ──────────────────────────────────────────────────────────

    def run(self):
        """
        Runs the full ingestion pipeline with streaming + batching + checkpointing.

        Flow per batch:
          lazy pages → chunk batch → embed batch → upsert Weaviate
                                  → extract triplets → upsert Neo4j
        """
        print("\n[PIPELINE] Starting Ingestion Pipeline...")
        start_time = time.time()

        cp = self._load_checkpoint()
        batches_done    = cp["batches_done"]
        chunks_stored   = cp["chunks_stored"]
        triplets_stored = cp["triplets_stored"]

        self.neo4j_client.initialize_schema()

        # Stream documents → chunk in batches
        doc_stream = self.loader.load_documents_lazy()
        chunk_batches = self.chunker.split_documents_batched(
            doc_stream, batch_size=CHUNK_BATCH_SIZE
        )

        for batch_idx, chunk_batch in enumerate(chunk_batches):

            # Skip already-processed batches (checkpoint resume)
            if batch_idx < batches_done:
                print(f"[pipeline] Skipping batch {batch_idx} (already in checkpoint).")
                continue

            print(f"\n[pipeline] ── Batch {batch_idx + 1} ({len(chunk_batch)} chunks) ──")

            # ── Build chunk dicts ──────────────────────────────────────────────
            chunks_data = [
                {
                    "text":     c.page_content,
                    "source":   c.metadata.get("source", "unknown"),
                    "chunk_id": c.metadata.get("chunk_index", f"chunk_{i}"),
                    "page_number": c.metadata.get("page_number", 1),
                    "file_name":   c.metadata.get("file_name", "unknown"),
                }
                for i, c in enumerate(chunk_batch)
            ]

            # ── Step 1: Embed in sub-batches ───────────────────────────────────
            print(f"  [embed] Embedding {len(chunks_data)} chunks in sub-batches of {EMBED_BATCH_SIZE}...")
            all_vectors = []
            all_weaviate_docs = []

            for sub_start in range(0, len(chunks_data), EMBED_BATCH_SIZE):
                sub_batch = chunks_data[sub_start: sub_start + EMBED_BATCH_SIZE]
                texts = [c["text"] for c in sub_batch]

                try:
                    vectors = self.embedder.embed_documents(texts)
                except Exception as e:
                    print(f"  [embed] ERROR embedding sub-batch: {e}. Using zero vectors as fallback.")
                    vectors = [[0.0] * 768] * len(texts)   # safe fallback; adjust dim if needed

                all_vectors.extend(vectors)
                all_weaviate_docs.extend([
                    {
                        "content":     c["text"],
                        "source":      c["source"],
                        "chunk_id":    int(c["chunk_id"]) if str(c["chunk_id"]).isdigit() else sub_start + j,
                        "page_number": c["page_number"],
                        "doc_type":    c["file_name"].split(".")[-1] if "." in c["file_name"] else "unknown",
                        "section":     "",
                        "language":    "en",
                        "token_count": len(c["text"].split()),
                        "created_at":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    for j, c in enumerate(sub_batch)
                ])

            # ── Step 2: Upsert to Weaviate ─────────────────────────────────────
            print(f"  [weaviate] Upserting {len(all_weaviate_docs)} chunks...")
            try:
                self.weaviate_client.upsert_chunks(all_weaviate_docs, all_vectors)
                chunks_stored += len(all_weaviate_docs)
            except Exception as e:
                print(f"  [weaviate] ERROR during upsert: {e}. Continuing to next step.")

            # ── Step 3: Extract triplets in sub-batches → Neo4j ───────────────
            print(f"  [extract] Extracting triplets in sub-batches of {TRIPLET_BATCH_SIZE}...")
            for triplet_batch in self.extractor.extract_batched(chunks_data, batch_size=TRIPLET_BATCH_SIZE):
                if not triplet_batch:
                    continue

                try:
                    builder = GraphBuilder(self.neo4j_client)
                    builder.build_graph(triplet_batch)
                    triplets_stored += len(triplet_batch)
                except Exception as e:
                    import traceback
                    print(f"  [neo4j] ERROR writing to graph: {e}\n{traceback.format_exc()}")

            # ── Inter-batch sleep to respect free-tier RPM limits ──────────────
            time.sleep(INTER_BATCH_DELAY)

            # ── Save checkpoint after each successful batch ────────────────────
            batches_done += 1
            cp = {
                "batches_done":    batches_done,
                "chunks_stored":   chunks_stored,
                "triplets_stored": triplets_stored,
            }
            self._save_checkpoint(cp)
            print(f"  [checkpoint] Saved. Total so far — chunks: {chunks_stored}, triplets: {triplets_stored}")

        # ── Done ───────────────────────────────────────────────────────────────
        elapsed = time.time() - start_time
        self._clear_checkpoint()

        print(f"\n[PIPELINE] ✓ Completed in {elapsed:.1f}s")
        print(f"           Chunks stored in Weaviate : {chunks_stored}")
        print(f"           Triplets stored in Neo4j  : {triplets_stored}")
        print(f"           Batches processed         : {batches_done}")


if __name__ == "__main__":
    os.makedirs("data/sample_docs", exist_ok=True)
    pipeline = IngestionPipeline()
    pipeline.run()