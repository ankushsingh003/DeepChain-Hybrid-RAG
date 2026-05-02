"""
DeepChain-Hybrid-RAG: Enterprise Knowledge Intelligence
Module: Weaviate Client — Production Grade

Fixes & Additions over v1:
  - Cloud + local connection with API key auth support
  - Deterministic UUID from source+chunk_id → idempotent upserts, no duplicates
  - Configurable HNSW index parameters (ef, efConstruction, maxConnections)
  - Binary Quantization (BQ) option for memory efficiency at scale
  - Rich metadata schema: doc_type, page_number, section, language, token_count, created_at
  - Multi-tenancy support (opt-in via env flag)
  - Failed-object tracking and retry in batch upsert
  - Configurable batch size for upsert
  - Pre-filter support in vector search (metadata filters)
  - Distance threshold in search to cut irrelevant results
  - Connection retry with backoff
  - Full logging instead of bare prints
"""

import os
import uuid
import hashlib
import logging
import time
from typing import List, Dict, Any, Optional

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import (
    Property,
    DataType,
    Configure,
    VectorDistances,
)
from weaviate.classes.query import Filter
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_COLLECTION = "DocumentChunk"
DEFAULT_UPSERT_BATCH = 200          # Objects per Weaviate batch call
DEFAULT_EF = 128                    # HNSW ef — higher = better recall, slower query
DEFAULT_EF_CONSTRUCTION = 200       # HNSW build quality
DEFAULT_MAX_CONNECTIONS = 64        # HNSW graph connectivity
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_DISTANCE_THRESHOLD = 0.30   # Reject results with cosine distance > this
MAX_UPSERT_RETRIES = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_deterministic_uuid(source: str, chunk_id: Any) -> str:
    """
    Derive a stable UUID from (source, chunk_id).
    Re-ingesting the same chunk will produce the same UUID → safe upsert, no duplicates.
    """
    key = f"{source}::{chunk_id}"
    hex_digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return str(uuid.UUID(hex_digest))


# ---------------------------------------------------------------------------
# WeaviateClient
# ---------------------------------------------------------------------------

class WeaviateNotAvailableError(RuntimeError):
    """Raised when Weaviate cannot be reached at startup."""
    pass


class WeaviateClient:
    """
    Production-grade Weaviate client for DeepChain-Hybrid-RAG.
    Raises WeaviateNotAvailableError (not a raw socket error) when Weaviate
    is unreachable, so the API layer can return a clean 503 to the frontend.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        upsert_batch_size: int = DEFAULT_UPSERT_BATCH,
        enable_multi_tenancy: bool = False,
        enable_bq: bool = False,
    ):
        self.collection_name = collection_name
        self.upsert_batch_size = upsert_batch_size
        self.enable_multi_tenancy = enable_multi_tenancy
        self.enable_bq = enable_bq

        self.client = self._connect_with_retry()
        self.create_schema()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """
        Quick TCP health-check — returns True if Weaviate is reachable.
        Call this BEFORE constructing WeaviateClient to give a clean error.
        """
        import socket
        host = os.getenv("WEAVIATE_HOST", "127.0.0.1")
        port = int(os.getenv("WEAVIATE_PORT", 8080))
        try:
            with socket.create_connection((host, port), timeout=3):
                return True
        except OSError:
            return False

    def _connect_with_retry(self, attempts: int = 3, delay: float = 2.0):
        """Connect to Weaviate (cloud or local) with retry/backoff.
        Raises WeaviateNotAvailableError with a clear message on failure."""
        weaviate_url = os.getenv("WEAVIATE_URL", "")
        api_key      = os.getenv("WEAVIATE_API_KEY", "")
        host         = os.getenv("WEAVIATE_HOST", "127.0.0.1")
        port         = int(os.getenv("WEAVIATE_PORT", 8080))

        for attempt in range(1, attempts + 1):
            try:
                if weaviate_url and "weaviate.network" in weaviate_url:
                    logger.info(f"[WeaviateClient] Connecting to WCS: {weaviate_url}")
                    client = weaviate.connect_to_weaviate_cloud(
                        cluster_url=weaviate_url,
                        auth_credentials=Auth.api_key(api_key) if api_key else None,
                    )
                else:
                    grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", 50051))
                    logger.info(f"[WeaviateClient] Connecting to local Weaviate at {host}:{port}")
                    client = weaviate.connect_to_local(
                        host=host,
                        port=port,
                        grpc_port=grpc_port,
                        additional_config=weaviate.config.AdditionalConfig(timeout=(20, 60)) # (connect, read) timeouts
                    )

                logger.info("[WeaviateClient] Connected successfully.")
                return client

            except Exception as e:
                if attempt == attempts:
                    logger.error(f"[WeaviateClient] Connection failed after {attempts} attempts: {e}")
                    raise WeaviateNotAvailableError(
                        f"Weaviate is not running or not reachable at {host}:{port}. "
                        f"Start it with: docker run -d -p 8080:8080 -p 50051:50051 "
                        f"cr.weaviate.io/semitechnologies/weaviate:latest"
                    ) from e
                logger.warning(f"[WeaviateClient] Attempt {attempt} failed ({e}). Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2

    def close(self):
        """Cleanly close the Weaviate connection."""
        self.client.close()
        logger.info("[WeaviateClient] Connection closed.")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def create_schema(self):
        """
        Creates the DocumentChunk collection if it doesn't exist.

        Schema fields:
            content       — the raw chunk text
            source        — file/document path or identifier
            chunk_id      — integer position within the source document
            doc_type      — e.g. "pdf", "html", "csv", "markdown"
            page_number   — page within the source document (if applicable)
            section       — section heading or logical group
            language      — ISO 639-1 language code, e.g. "en"
            token_count   — approximate token length of the chunk
            created_at    — ISO 8601 ingestion timestamp (stored as text for portability)

        Index:
            HNSW with tuned ef/efConstruction/maxConnections for large-scale recall.
            Optional Binary Quantization for memory-efficient storage.
        """
        if self.client.collections.exists(self.collection_name):
            logger.info(f"[WeaviateClient] Collection '{self.collection_name}' already exists. Skipping creation.")
            return

        logger.info(f"[WeaviateClient] Creating collection '{self.collection_name}'...")

        # Build vector index config
        if self.enable_bq:
            vector_index_config = Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                ef=DEFAULT_EF,
                ef_construction=DEFAULT_EF_CONSTRUCTION,
                max_connections=DEFAULT_MAX_CONNECTIONS,
                quantizer=Configure.VectorIndex.Quantizer.bq(),
            )
        else:
            vector_index_config = Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                ef=DEFAULT_EF,
                ef_construction=DEFAULT_EF_CONSTRUCTION,
                max_connections=DEFAULT_MAX_CONNECTIONS,
            )

        # Multi-tenancy config
        mt_config = Configure.multi_tenancy(enabled=True) if self.enable_multi_tenancy else None

        create_kwargs = dict(
            name=self.collection_name,
            properties=[
                Property(name="content",     data_type=DataType.TEXT),
                Property(name="source",      data_type=DataType.TEXT),
                Property(name="chunk_id",    data_type=DataType.INT),
                Property(name="doc_type",    data_type=DataType.TEXT),
                Property(name="page_number", data_type=DataType.INT),
                Property(name="section",     data_type=DataType.TEXT),
                Property(name="language",    data_type=DataType.TEXT),
                Property(name="token_count", data_type=DataType.INT),
                Property(name="created_at",  data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.none(),   # External embeddings
            vector_index_config=vector_index_config,
        )

        if mt_config is not None:
            create_kwargs["multi_tenancy_config"] = mt_config

        self.client.collections.create(**create_kwargs)
        logger.info(f"[WeaviateClient] Collection '{self.collection_name}' created.")

    def delete_collection(self):
        """Drop and recreate the collection. Useful for full re-indexing."""
        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)
            logger.info(f"[WeaviateClient] Collection '{self.collection_name}' deleted.")
        self.create_schema()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        vectors: List[List[float]],
        tenant: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Idempotent batch upsert of text chunks + pre-computed vectors.

        Each chunk dict should contain at minimum:
            source (str), chunk_id (int), content (str)

        Optional enrichment fields (used if present):
            doc_type, page_number, section, language, token_count, created_at

        Args:
            chunks:   List of property dicts for each chunk.
            vectors:  Corresponding embedding vectors (same length as chunks).
            tenant:   Tenant name for multi-tenant collections (optional).

        Returns:
            {"inserted": N, "failed": M}
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must have the same length."
            )

        collection = self.client.collections.get(self.collection_name)
        total = len(chunks)
        total_batches = (total + self.upsert_batch_size - 1) // self.upsert_batch_size
        inserted = 0
        failed_objects = []

        logger.info(f"[WeaviateClient] Upserting {total} chunks in {total_batches} batches...")

        for batch_idx in range(total_batches):
            start = batch_idx * self.upsert_batch_size
            end = min(start + self.upsert_batch_size, total)
            batch_chunks = chunks[start:end]
            batch_vectors = vectors[start:end]

            for attempt in range(1, MAX_UPSERT_RETRIES + 1):
                try:
                    batch_context = (
                        collection.with_tenant(tenant).batch.dynamic()
                        if tenant
                        else collection.batch.dynamic()
                    )
                    with batch_context as batch:
                        for chunk, vector in zip(batch_chunks, batch_vectors):
                            # Deterministic UUID → safe re-ingestion
                            obj_uuid = _make_deterministic_uuid(
                                chunk.get("source", "unknown"),
                                chunk.get("chunk_id", 0),
                            )
                            batch.add_object(
                                properties=chunk,
                                vector=vector,
                                uuid=obj_uuid,
                            )

                    # Check for silent failures inside this batch
                    if hasattr(batch, "failed_objects") and batch.failed_objects:
                        count = len(batch.failed_objects)
                        logger.error(
                            f"[WeaviateClient] Batch {batch_idx + 1}: {count} objects failed."
                        )
                        for fo in batch.failed_objects:
                            logger.error(f"  Failed object: {fo}")
                        failed_objects.extend(batch.failed_objects)

                    inserted += len(batch_chunks) - len(
                        batch.failed_objects if hasattr(batch, "failed_objects") and batch.failed_objects else []
                    )
                    break  # success

                except Exception as e:
                    if attempt == MAX_UPSERT_RETRIES:
                        logger.error(
                            f"[WeaviateClient] Batch {batch_idx + 1} failed after {attempt} retries: {e}"
                        )
                        failed_objects.append({"batch": batch_idx + 1, "error": str(e)})
                    else:
                        wait = 2 ** attempt
                        logger.warning(
                            f"[WeaviateClient] Batch {batch_idx + 1} attempt {attempt} failed ({e}). "
                            f"Retrying in {wait}s..."
                        )
                        time.sleep(wait)

            logger.info(
                f"[WeaviateClient] Progress: {min(end, total)}/{total} chunks processed."
            )

        logger.info(
            f"[WeaviateClient] Upsert complete. inserted={inserted}, failed={len(failed_objects)}"
        )
        return {"inserted": inserted, "failed": len(failed_objects)}

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        vector: List[float],
        limit: int = DEFAULT_SEARCH_LIMIT,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        filters: Optional[Any] = None,
        return_properties: Optional[List[str]] = None,
        tenant: Optional[str] = None,
    ):
        """
        Vector similarity search with distance threshold and optional metadata filters.

        Args:
            vector:             Query embedding vector.
            limit:              Max number of results to return.
            distance_threshold: Cosine distance ceiling — results beyond this are dropped.
            filters:            Weaviate Filter object for metadata pre-filtering (optional).
            return_properties:  Which fields to fetch (defaults to all schema fields).
            tenant:             Tenant name for multi-tenant collections (optional).

        Returns:
            List of Weaviate result objects (with .properties and .metadata.distance).
        """
        if return_properties is None:
            return_properties = [
                "content", "source", "chunk_id",
                "doc_type", "page_number", "section",
                "language", "token_count", "created_at",
            ]

        collection = self.client.collections.get(self.collection_name)
        if tenant:
            collection = collection.with_tenant(tenant)

        query_kwargs = dict(
            near_vector=vector,
            limit=limit,
            distance=distance_threshold,
            return_properties=return_properties,
            return_metadata=["distance"],
        )
        if filters is not None:
            query_kwargs["filters"] = filters

        response = collection.query.near_vector(**query_kwargs)
        return response.objects

    def count(self, tenant: Optional[str] = None) -> int:
        """Return total number of objects in the collection."""
        collection = self.client.collections.get(self.collection_name)
        if tenant:
            collection = collection.with_tenant(tenant)
        result = collection.aggregate.over_all(total_count=True)
        return result.total_count

    # ------------------------------------------------------------------
    # Tenant management (multi-tenancy)
    # ------------------------------------------------------------------

    def create_tenant(self, tenant_name: str):
        """Create a new tenant in the collection (requires multi_tenancy enabled)."""
        from weaviate.classes.tenants import Tenant
        collection = self.client.collections.get(self.collection_name)
        collection.tenants.create([Tenant(name=tenant_name)])
        logger.info(f"[WeaviateClient] Tenant '{tenant_name}' created.")

    def list_tenants(self) -> List[str]:
        """List all tenant names in the collection."""
        collection = self.client.collections.get(self.collection_name)
        tenants = collection.tenants.get()
        return [t.name for t in tenants.values()]


# ---------------------------------------------------------------------------
# Convenience filter builders (re-exported for use in retriever)
# ---------------------------------------------------------------------------

def filter_by_doc_type(doc_type: str) -> Filter:
    return Filter.by_property("doc_type").equal(doc_type)

def filter_by_language(language: str) -> Filter:
    return Filter.by_property("language").equal(language)

def filter_by_source(source: str) -> Filter:
    return Filter.by_property("source").equal(source)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        client = WeaviateClient()
        count = client.count()
        print(f"[SUCCESS] Connected. Collection has {count} objects.")
        client.close()
    except Exception as e:
        print(f"[ERROR] {e}")