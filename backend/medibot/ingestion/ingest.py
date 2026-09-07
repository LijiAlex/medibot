"""Step 6 of ingestion: the orchestrator. Runs the whole pipeline and writes Qdrant.

    python -m medibot.ingestion.ingest                 # all 12 documents
    python -m medibot.ingestion.ingest --only drug_formulary.pdf

Per document: load -> parse -> chunk -> prepare -> delete its old points -> upsert.
Deleting first (by source_document) handles the case where a re-run produces
fewer chunks than before; deterministic point ids alone would leave stale ones
behind. Deterministic ids still matter: they make the upsert overwrite rather
than duplicate when the chunk count is unchanged.

Finishes with verify(): counts per collection and an RBAC probe that must return
zero restricted chunks for a nurse. Ingestion that fails verification is a bug,
not a warning.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

from medibot.config import COLLECTION_ROLES, QDRANT_COLLECTION
from medibot.ingestion.chunker import chunk_document
from medibot.ingestion.loader import SourceDoc, load_sources
from medibot.ingestion.parser import parse
from medibot.ingestion.prepare import prepare
from medibot.vectorstore import KEY_COLLECTION, get_client, get_vectorstore, rbac_filter, source_filter

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    """What one ingestion run produced, for logging and tests."""

    documents: int = 0
    chunks: int = 0
    per_document: dict[str, int] = field(default_factory=dict)
    per_collection: dict[str, int] = field(default_factory=dict)
    total_points: int = 0
    rbac_probe_ok: bool = False
    seconds: float = 0.0


def ingest_document(source: SourceDoc, vectorstore, client: QdrantClient) -> int:
    """Run steps 2-6 for one file and return the number of points written."""
    doc = parse(source)
    docs = prepare(chunk_document(doc), source, doc)
    client.delete(
        collection_name=vectorstore.collection_name,
        points_selector=models.FilterSelector(filter=source_filter(source.source_document)),
    )
    vectorstore.add_documents(docs, ids=[d.metadata["point_id"] for d in docs])
    return len(docs)


def count_per_collection(client: QdrantClient, name: str = QDRANT_COLLECTION) -> dict[str, int]:
    """Exact point counts per collection, straight from Qdrant."""
    out: dict[str, int] = {}
    for coll in COLLECTION_ROLES:
        flt = models.Filter(must=[models.FieldCondition(key=KEY_COLLECTION, match=models.MatchValue(value=coll))])
        out[coll] = client.count(collection_name=name, count_filter=flt, exact=True).count
    return out


def rbac_probe(vectorstore, role: str = "nurse", query: str = "insurance billing codes and equipment fault codes") -> bool:
    """Ask, as `role`, for content that role must never see. True if nothing leaks.

    The query deliberately names restricted topics. With the filter applied at
    the Qdrant level the results can only come from the role's own collections,
    whatever the words in the query.
    """
    allowed = {c for c, roles in COLLECTION_ROLES.items() if role in roles}
    hits = vectorstore.similarity_search(query, k=10, filter=rbac_filter(role))
    leaked = [h.metadata["collection"] for h in hits if h.metadata["collection"] not in allowed]
    if leaked:
        log.error("RBAC probe as %s leaked collections: %s", role, leaked)
    return not leaked


def verify(client: QdrantClient, vectorstore, report: IngestReport) -> IngestReport:
    """Fill the report with what Qdrant actually holds and run the RBAC probe."""
    report.per_collection = count_per_collection(client, vectorstore.collection_name)
    report.total_points = client.count(collection_name=vectorstore.collection_name, exact=True).count
    report.rbac_probe_ok = rbac_probe(vectorstore)
    return report


def ingest(sources: list[SourceDoc] | None = None, client: QdrantClient | None = None) -> IngestReport:
    """Ingest the given sources (default: everything under data/) and verify."""
    t0 = time.time()
    client = client or get_client()
    vectorstore = get_vectorstore(client)
    report = IngestReport()
    for src in sources or load_sources():
        n = ingest_document(src, vectorstore, client)
        report.per_document[src.source_document] = n
        report.documents += 1
        report.chunks += n
        log.info("%-30s %3d chunks  [%s]", src.source_document, n, src.collection)
    verify(client, vectorstore, report)
    report.seconds = time.time() - t0
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest MediBot documents into Qdrant.")
    ap.add_argument("--only", metavar="FILENAME", help="ingest a single source document, e.g. drug_formulary.pdf")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("docling", "transformers", "sentence_transformers", "httpx", "RapidOCR"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    sources = load_sources()
    if args.only:
        sources = [s for s in sources if s.source_document == args.only]
        if not sources:
            raise SystemExit(f"no source named {args.only!r}")

    r = ingest(sources)
    print(f"\ndocuments={r.documents} chunks_written={r.chunks} total_points={r.total_points} in {r.seconds:.0f}s")
    print("per collection:", r.per_collection)
    print("RBAC probe (nurse must see no billing/equipment):", "OK" if r.rbac_probe_ok else "FAILED")
    if not r.rbac_probe_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
