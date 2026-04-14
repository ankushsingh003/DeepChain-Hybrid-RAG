"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Data Ingestion Pipeline Orchestrator
"""

from typing import List
from ingestion.loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.extractor import GraphExtractor, KnowledgeGraph

class IngestionPipeline:
    def __init__(self, data_path: str = "data/sample_docs"):
        self.loader = DocumentLoader(data_path)
        self.chunker = DocumentChunker()
        self.extractor = GraphExtractor()

    def run(self) -> KnowledgeGraph:
        """Runs the full ingestion pipeline: Load -> Chunk -> Extract Knowledge."""
        print("\n[PIPELINE] Starting Information Extraction Pipeline...")
        
        # 1. Load
        documents = self.loader.load_documents()
        if not documents:
            print("[!] No documents found. Exiting.")
            return KnowledgeGraph(entities=[], relationships=[])

        # 2. Chunk
        chunks = self.chunker.split_documents(documents)
        
        # 3. Extract (Batching chunks into extraction)
        master_kg = KnowledgeGraph(entities=[], relationships=[])
        
        # Note: In a production system, we would use async/parallel extraction here.
        # For simplicity, we iterate through chunks.
        for i, chunk in enumerate(chunks):
            print(f"\n[PIPELINE] Processing Chunk {i+1}/{len(chunks)}...")
            try:
                chunk_kg = self.extractor.extract(chunk.page_content)
                master_kg.entities.extend(chunk_kg.entities)
                master_kg.relationships.extend(chunk_kg.relationships)
            except Exception as e:
                print(f"[!] Error processing chunk {i+1}: {e}")

        # Deduplicate entities (Simple name-based)
        seen_entities = set()
        deduped_entities = []
        for e in master_kg.entities:
            if e.name.lower() not in seen_entities:
                deduped_entities.append(e)
                seen_entities.add(e.name.lower())
        
        master_kg.entities = deduped_entities
        
        print(f"\n[PIPELINE] Completed.")
        print(f"[+] Total Unique Entities: {len(master_kg.entities)}")
        print(f"[+] Total Relationships: {len(master_kg.relationships)}")
        
        return master_kg

if __name__ == "__main__":
    # Ensure sample data exists
    import os
    os.makedirs("data/sample_docs", exist_ok=True)
    sample_path = "data/sample_docs/financial_news.txt"
    if not os.path.exists(sample_path):
        with open(sample_path, "w") as f:
            f.write("Google Inc. announced a partnership with Anthropic AI. "
                    "The deal is valued at $2 billion and aims to accelerate cloud computing.")

    pipeline = IngestionPipeline()
    final_kg = pipeline.run()
    
    # Save processed KG for reference
    os.makedirs("data/processed", exist_ok=True)
    with open("data/processed/extracted_kg.json", "w") as f:
        f.write(json.dumps([e.dict() for e in final_kg.entities], indent=2))
