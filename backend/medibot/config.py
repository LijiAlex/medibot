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
# openai/gpt-oss-120b via Groq. Was gpt-oss-20b until 2026-09-10: with the 20b model,
# create_sql_query_chain returned an empty string on about a quarter of calls, because it
# stops generation at "\nSQLResult:" and the 20b model sometimes begins its visible output
# at that line. Measured on the same prompt and question, 120b was usable 5/5 with the
# stop token and 20b 3/4 without it. Groq's daily token cap is per model, so this also
# came with a fresh budget.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

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
#
# Was 0.0, argued from the ms-marco cross-encoder's own decision boundary and checked
# against seven questions. That sample had a hole: a doctor asking about ECG
# interpretation flags scored -1.11 and was refused an answer that had been retrieved at
# rank 1. Measured again over 16 answerable and 8 blocked questions on 2026-09-12:
#
#   answerable   worst -1.11, then +0.27, +1.74, +2.11 ... +9.26
#   blocked      best  -7.99, then -9.46, -10.22 ... -11.07
#
# The two groups are 6.88 apart, so this sits in the middle of that gap rather than on
# the model's nominal boundary. It is an empirical number and tests/test_hybrid_rag.py
# pins the separation so it cannot drift back into the answerable range.
RELEVANCE_THRESHOLD = -4.0

# How far the classified collection must beat the best one a role may read before the
# refusal names it. Below this the two are too close to call, and the honest reply is
# that nothing was found rather than that access was denied. Measured 2026-09-11:
# refusals that were clearly right had gaps of 0.058 and above; a technician asking about
# ventilator maintenance produced a gap of 0.019 between nursing and equipment.
CLASSIFY_MARGIN = 0.04
