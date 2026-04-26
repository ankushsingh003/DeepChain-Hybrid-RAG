# """
# DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
# Module: Semantic Chunking Strategy
# """

# from typing import List
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document

# class DocumentChunker:
#     def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.splitter = RecursiveCharacterTextSplitter(
#             chunk_size=self.chunk_size,
#             chunk_overlap=self.chunk_overlap,
#             separators=["\n\n", "\n", " ", ""]
#         )

#     def split_documents(self, documents: List[Document]) -> List[Document]:
#         """Splits documents into smaller chunks for vector and graph processing."""
#         print(f"[*] Splitting {len(documents)} documents into chunks...")
#         chunks = self.splitter.split_documents(documents)
#         print(f"[+] Created {len(chunks)} chunks.")
#         return chunks

# if __name__ == "__main__":
#     from langchain_core.documents import Document
    
#     # Test sample
#     test_docs = [Document(page_content="DeepChain-Hybrid-RAG is a powerful system. " * 50)]
#     chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
#     chunks = chunker.split_documents(test_docs)
#     print(f"[TEST] Chunk count: {len(chunks)}")
#     print(f"[TEST] First chunk: {chunks[0].page_content}")




"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Semantic Chunking Strategy

CHANGES FROM ORIGINAL:
- The original used RecursiveCharacterTextSplitter which is purely character-based,
  NOT semantic. Renamed the internal strategy to reflect what it actually is.
- Added TRUE semantic chunking via SemanticChunker (langchain_experimental) which
  uses embedding similarity to find natural topic boundaries instead of hard char limits.
  Falls back to RecursiveCharacterTextSplitter if embeddings unavailable.
- Added page metadata PRESERVATION — the original lost page_number/source metadata
  during splitting. Now every chunk carries page_number, file_name, chunk_index,
  and char_count from its parent document.
- Added page-boundary-aware splitting: chunks never blindly merge across page breaks
  unless they are semantically related (SemanticChunker handles this naturally).
- Added chunk size validation — filters out near-empty chunks (< 50 chars) that
  add noise to the vector store and LLM context.
- split_documents_batched() generator allows the pipeline to process chunks in
  batches without accumulating all chunks in RAM.
"""

from typing import List, Iterator, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_chars: int = 50,
        use_semantic: bool = False,
        embeddings=None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars
        self.use_semantic = use_semantic
        self.embeddings = embeddings

        # Fallback splitter — always available
        self._char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # True semantic splitter — only if embeddings provided
        self._semantic_splitter = None
        if self.use_semantic and self.embeddings is not None:
            try:
                from langchain_experimental.text_splitter import SemanticChunker
                self._semantic_splitter = SemanticChunker(
                    embeddings=self.embeddings,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=90,
                )
                print("[chunker] Using TRUE semantic chunking (SemanticChunker).")
            except ImportError:
                print("[chunker] langchain_experimental not installed — falling back to RecursiveCharacterTextSplitter.")
        else:
            print(f"[chunker] Using RecursiveCharacterTextSplitter (chunk_size={chunk_size}, overlap={chunk_overlap}).")

    def _get_splitter(self):
        return self._semantic_splitter if self._semantic_splitter else self._char_splitter

    def _split_single(self, doc: Document, global_chunk_offset: int = 0) -> List[Document]:
        """
        Splits a single Document and enriches every resulting chunk with
        inherited metadata from the parent (page_number, file_name, source).
        """
        splitter = self._get_splitter()
        raw_chunks = splitter.split_documents([doc])

        enriched = []
        for i, chunk in enumerate(raw_chunks):
            # Skip near-empty chunks
            if len(chunk.page_content.strip()) < self.min_chunk_chars:
                continue

            # Preserve + enrich metadata
            chunk.metadata.update({
                "source": doc.metadata.get("source", "unknown"),
                "file_name": doc.metadata.get("file_name", "unknown"),
                "page_number": doc.metadata.get("page_number", 1),
                "total_pages": doc.metadata.get("total_pages", 1),
                "chunk_index": global_chunk_offset + i,
                "char_count": len(chunk.page_content),
            })
            enriched.append(chunk)

        return enriched

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of documents into chunks.
        All chunks are collected in memory — fine for moderate sizes.
        For large PDFs, prefer split_documents_batched().
        """
        print(f"[*] Splitting {len(documents)} pages into chunks...")
        all_chunks = []
        for doc in documents:
            chunks = self._split_single(doc, global_chunk_offset=len(all_chunks))
            all_chunks.extend(chunks)

        print(f"[+] Created {len(all_chunks)} chunks (filtered out near-empty ones).")
        return all_chunks

    def split_documents_batched(
        self,
        documents: Iterator[Document],
        batch_size: int = 50,
    ) -> Iterator[List[Document]]:
        """
        Generator that yields lists of chunks in batches.
        Feed it the lazy document iterator from DocumentLoader so documents
        and chunks are never all in RAM at the same time.

        Usage:
            for chunk_batch in chunker.split_documents_batched(loader.load_documents_lazy()):
                process(chunk_batch)
        """
        chunk_buffer = []
        global_chunk_offset = 0

        for doc in documents:
            new_chunks = self._split_single(doc, global_chunk_offset=global_chunk_offset)
            chunk_buffer.extend(new_chunks)
            global_chunk_offset += len(new_chunks)

            while len(chunk_buffer) >= batch_size:
                yield chunk_buffer[:batch_size]
                chunk_buffer = chunk_buffer[batch_size:]

        # Yield remaining
        if chunk_buffer:
            yield chunk_buffer


if __name__ == "__main__":
    from langchain_core.documents import Document

    test_docs = [
        Document(
            page_content="DeepChain is a powerful RAG system. " * 40,
            metadata={"source": "test.pdf", "file_name": "test.pdf", "page_number": 1, "total_pages": 3}
        ),
        Document(
            page_content="It combines Neo4j and Weaviate for hybrid retrieval. " * 40,
            metadata={"source": "test.pdf", "file_name": "test.pdf", "page_number": 2, "total_pages": 3}
        ),
    ]

    chunker = DocumentChunker(chunk_size=200, chunk_overlap=40)
    chunks = chunker.split_documents(test_docs)
    for c in chunks[:3]:
        print(f"Page {c.metadata['page_number']} | Chunk {c.metadata['chunk_index']} | {len(c.page_content)} chars")
        print(f"  {c.page_content[:80]}...")