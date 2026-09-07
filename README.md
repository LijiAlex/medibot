# MediBot

Role-based Hybrid RAG + SQL RAG assistant for a hospital network. Codebasics AI Engineering Bootcamp, Assignment 2.

**Status: Part 1 (ingestion) complete. Part 2 (API, retrieval, SQL RAG, UI) in progress.**

## Layout

```
data/       source documents and SQLite DB, exactly as provided (read-only)
store/      everything preparation produces and the app reads
            store/qdrant/   embedded Qdrant, 283 points (rebuilt by ingestion, not committed)
            store/sql/      schema context + example queries for SQL RAG (coming)
backend/    Python 3.12, uv. medibot/ingestion = Part 1, medibot/retrieval + api = Part 2
frontend/   Next.js (coming)
```

## Ingestion pipeline

load → parse (pre-process · Docling with heading levels · post-process) → HybridChunker → prepare (breadcrumb + metadata + deterministic ids) → dense (MiniLM) + sparse (BM25) → Qdrant, verified with an RBAC probe.

```bash
cd backend
uv sync
uv run python -m medibot.ingestion.ingest      # ~90 s, writes ../store/qdrant
uv run pytest                                   # 35 tests
```

Full design notes, decisions and measurements are being written up; README will be completed with setup, architecture diagram, adversarial RBAC examples and tool substitutions at submission.
