"""Qdrant access shared by ingestion (writes) and retrieval (reads).

One collection, two named vectors per point:
  dense   384-d cosine, from MiniLM
  sparse  BM25 weights, from FastEmbed, with Qdrant's IDF modifier
plus a payload: page_content + metadata (the five required fields and extras).

Embedded Qdrant (path=...) runs inside the Python process with no server. Only
one process can hold the store open at a time, so ingestion and the API must
not run simultaneously against the same path. Switching to a Qdrant server
later is one change in get_client().

RBAC lives here too: rbac_filter(role) is the Qdrant-side filter every retrieval
query must pass, so restricted chunks are never returned to the application.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from medibot.config import DENSE_DIM, QDRANT_COLLECTION, QDRANT_PATH
from medibot.ingestion.embeddings import get_dense, get_sparse

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# Payload keys as LangChain's QdrantVectorStore writes them.
META = QdrantVectorStore.METADATA_KEY  # "metadata"
KEY_ACCESS_ROLES = f"{META}.access_roles"
KEY_COLLECTION = f"{META}.collection"
KEY_SOURCE_DOCUMENT = f"{META}.source_document"


@lru_cache(maxsize=4)
def get_client(path: Path = QDRANT_PATH) -> QdrantClient:
    """Embedded Qdrant client, one per path per process."""
    return QdrantClient(path=str(path))


def ensure_collection(client: QdrantClient, name: str = QDRANT_COLLECTION) -> bool:
    """Create the collection with named dense + sparse vectors if it does not exist.

    Returns True if it was created now, False if it already existed. Idempotent,
    so ingestion can call it on every run.

    No payload indexes: embedded Qdrant ignores them, and at a few hundred points
    filtering by scan is instant. On a Qdrant server at scale, add a keyword index
    on metadata.access_roles so the RBAC filter becomes a lookup (see README).
    """
    if client.collection_exists(name):
        return False
    client.create_collection(
        collection_name=name,
        vectors_config={DENSE_VECTOR: models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    return True


def get_vectorstore(client: QdrantClient | None = None, name: str = QDRANT_COLLECTION) -> QdrantVectorStore:
    """LangChain handle over the collection, in HYBRID mode.

    HYBRID means: one query -> Qdrant runs dense and sparse search server-side,
    fuses the two rankings with RRF, returns one list. The same dense/sparse
    factories used at ingestion are wired in here, so query vectors land in the
    same space as the indexed ones.
    """
    client = client or get_client()
    ensure_collection(client, name)
    return QdrantVectorStore(
        client=client,
        collection_name=name,
        embedding=get_dense(),
        sparse_embedding=get_sparse(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR,
        sparse_vector_name=SPARSE_VECTOR,
    )


def rbac_filter(role: str) -> models.Filter:
    """Qdrant filter: only points whose access_roles list contains this role.

    Applied at query time, before similarity is even computed, so a nurse's
    search physically cannot return a billing chunk. This is the assignment's
    "enforced at the retrieval layer" requirement in one expression.
    """
    return models.Filter(must=[models.FieldCondition(key=KEY_ACCESS_ROLES, match=models.MatchAny(any=[role]))])


def source_filter(source_document: str) -> models.Filter:
    """Qdrant filter: all points that came from one source file."""
    return models.Filter(must=[models.FieldCondition(key=KEY_SOURCE_DOCUMENT, match=models.MatchValue(value=source_document))])
