"""Step 5 of ingestion, and shared with retrieval: the two embedding models.

Hybrid RAG stores two vectors per chunk (Session 5 pattern):

  dense   all-MiniLM-L6-v2 via HuggingFaceEmbeddings. 384 floats, L2-normalised,
          cosine similarity. Captures meaning: "full bars but no websites" finds
          the connectivity section without sharing a word with it.
  sparse  Qdrant/bm25 via FastEmbedSparse. A few non-zero weights on exact
          tokens. Captures identifiers: "IP-200", "F-05", "I21.4", "APN".

Both are built once per process and reused. Retrieval MUST import these same
factories: a query embedded by a different model (or the same model without
normalisation) lives in a different vector space and will not match the index.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse

from medibot.config import DENSE_MODEL, SPARSE_MODEL


@lru_cache(maxsize=1)
def get_dense() -> HuggingFaceEmbeddings:
    """Dense (semantic) embedder, loaded once.

    normalize_embeddings=True makes every vector unit length, so cosine
    similarity equals the dot product and scores are comparable across queries.
    CPU is enough: the model is ~22M parameters and we embed a few hundred chunks.
    """
    return HuggingFaceEmbeddings(
        model_name=DENSE_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_sparse() -> FastEmbedSparse:
    """Sparse (BM25 keyword) embedder, loaded once.

    FastEmbed ships Qdrant's BM25 implementation as a local model, no server
    needed. Output is a SparseVector: token indices plus weights, mostly zeros
    implied. batch_size=32 is the Session 5 setting; it only affects speed.
    """
    return FastEmbedSparse(model_name=SPARSE_MODEL, batch_size=32)
