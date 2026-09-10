"""Single source of truth shared by ingestion (Part 1) and the app (Part 2).

Anything that both sides must agree on lives here: role matrix, model names,
collection name, chunking limits, paths. Change it in one place only.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]



def _path(env_var: str, default: Path) -> Path:
    """Path from .env, or default. Relative values are anchored to REPO_ROOT, not to
    the shell's cwd, so `STORE_DIR=./store` means the same thing from backend/ or /."""
    value = os.getenv(env_var)
    return (REPO_ROOT / value).resolve() if value else default


# data/  : inputs exactly as given. Never modified.
DATA_DIR = _path("DATA_DIR", REPO_ROOT / "data")
DB_PATH = _path("DB_PATH", DATA_DIR / "db" / "mediassist.db")

# store/ : everything preparation produces and the app reads. Wipe + re-run prep rebuilds it.
STORE_DIR = _path("STORE_DIR", REPO_ROOT / "store")
QDRANT_PATH = STORE_DIR / "qdrant"                    # embedded Qdrant, gitignored

# --- Roles & collections ---------------------------------------------------
# Folder name under DATA_DIR == collection name. Source: assignment "Data Sources" table.
COLLECTION_ROLES: dict[str, list[str]] = {
    "general": ["doctor", "nurse", "billing_executive", "technician", "admin"],
    "clinical": ["doctor", "admin"],
    "nursing": ["nurse", "doctor", "admin"],
    "billing": ["billing_executive", "admin"],
    "equipment": ["technician", "admin"],
}

ROLES: list[str] = ["doctor", "nurse", "billing_executive", "technician", "admin"]

# Inverse view, derived once: role -> collections it may read. Used by /collections/{role}.
ROLE_COLLECTIONS: dict[str, list[str]] = {
    role: [c for c, roles in COLLECTION_ROLES.items() if role in roles] for role in ROLES
}

SQL_RAG_ROLES: list[str] = ["billing_executive", "admin"]

# --- Vector store ----------------------------------------------------------
QDRANT_COLLECTION = "medibot"

# --- Models ----------------------------------------------------------------
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-d, cosine, 256-token limit
DENSE_DIM = 384
SPARSE_MODEL = "Qdrant/bm25"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")  # via Groq

# --- Chunking --------------------------------------------------------------
# EMBED_MAX_TOKENS: DENSE_MODEL's max sequence length. Anything longer is silently
# truncated at embedding time, so the *final* embedded text must stay under it.
# HybridChunker measures oversized-item splits (tables) on body text only, and the
# prepare step prepends the document title on top of the heading path. Both add
# tokens after the chunker has done its work, so the chunker gets a smaller cap.
EMBED_MAX_TOKENS = 256
HEADING_HEADROOM = 32
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", str(EMBED_MAX_TOKENS - HEADING_HEADROOM)))  # 224

# --- Retrieval -------------------------------------------------------------
RETRIEVE_K = 10   # broad hybrid candidate set
RERANK_TOP_N = 3  # what actually reaches the LLM

# Below this top-1 rerank score, nothing retrieved is relevant and the LLM is not called.
# Not a tuned number: the ms-marco cross-encoder emits a relevance logit whose own
# decision boundary is zero. Measured on this corpus, the worst answerable question
# scored +2.57 and the best blocked one −7.99.
RELEVANCE_THRESHOLD = 0.0
