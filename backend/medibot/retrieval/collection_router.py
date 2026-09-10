"""Which collection does a question belong to? Used only to explain a refusal.

    utterances()            -> {collection: [section titles]}   drawn from the ingested corpus
    classify(question)      -> collection name, or None when nothing matches

This runs on one path only: after the score gate in hybrid_rag has decided that nothing
relevant was retrieved. It never decides a refusal, so a misclassification produces a
worse explanation, never a wrongly blocked question. That is why it sits after retrieval
rather than in front of it.

Utterances are the real section_title values already in Qdrant, minus the generic ones
("Introduction", "Note"), so the classifier is anchored to the corpus that was ingested
rather than to phrases someone invented.
"""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.routers import SemanticRouter

from medibot.config import DENSE_MODEL, QDRANT_COLLECTION
from medibot.vectorstore import get_client

# "1. ", "1.1 ", "5 ", "A. " at the start of a heading. The trailing dot is optional
# because "1.1 Pre-authorisation timeline" carries none, and the whitespace is required
# so that a real title like "24-hour urine collection" keeps its number.
_NUMBERING = re.compile(r"^(?:\d+(?:\.\d+)*|[A-Z])[.)]?\s+")

# Headings that appear under several collections and carry no topic of their own.
GENERIC_TITLES: frozenset[str] = frozenset({
    "introduction", "note", "see also", "best practice", "conventions", "declaration",
    "escalate", "frequency", "indications", "documentation requirements", "authorisation",
    "purpose", "scope", "overview", "summary", "references", "appendix", "contact",
    "critical values", "equipment checklist", "initial settings", "alarm response",
})

# Fits the thresholds below. Held-out questions live in tests/test_collection_router.py.
TRAIN: list[tuple[str, str | None]] = [
    ("Which ICD code do I use for sepsis billing?", "billing"),
    ("What is the room rent sub-limit for a private ward?", "billing"),
    ("Which insurers are empanelled with us?", "billing"),
    ("First-line antibiotic for pneumonia", "clinical"),
    ("What is the potassium reference range?", "clinical"),
    ("How do I interpret an ABG result?", "clinical"),
    ("How often do I change a peripheral cannula?", "nursing"),
    ("Steps for isolating a patient with MRSA", "nursing"),
    ("What are the ventilator alarm defaults?", "equipment"),
    ("Preventive maintenance schedule for the X-ray unit", "equipment"),
    ("What is the dress code policy?", "general"),
    ("How do I apply for maternity leave?", "general"),
    ("What is the weather today?", None),
    ("Tell me a joke", None),
]

# Output of refit() on 2026-09-10. Section titles are short phrases and questions are
# long, so cosine similarity runs lower than semantic-router's 0.5 default: at 0.5 the
# held-out set scored 5/12, every miss being "no route at all". Pinned because fit() is
# a random search and would otherwise give a different classifier in every process.
FITTED_THRESHOLDS: dict[str, float] = {
    "billing": 0.42,
    "clinical": 0.37,
    "equipment": 0.31,
    "general": 0.33,
    "nursing": 0.31,
}


def _clean(title: str) -> str:
    return _NUMBERING.sub("", title).strip()


@lru_cache(maxsize=1)
def utterances() -> dict[str, list[str]]:
    """Section titles per collection, straight out of the vector store.

    Scrolls the whole collection once at startup (payloads only, no vectors). Titles that
    are generic or a single word are dropped: they appear under several collections and
    would pull every question towards whichever route holds the most of them.
    """
    found: dict[str, set[str]] = defaultdict(set)
    offset = None
    while True:
        points, offset = get_client().scroll(
            QDRANT_COLLECTION, limit=500, offset=offset, with_payload=True, with_vectors=False
        )
        for point in points:
            meta = point.payload.get("metadata", point.payload)
            found[meta["collection"]].add(meta["section_title"])
        if offset is None:
            break
    return {
        collection: sorted(
            title
            for raw in titles
            if (title := _clean(raw)) and title.lower() not in GENERIC_TITLES and len(title.split()) >= 2
        )
        for collection, titles in found.items()
    }


@lru_cache(maxsize=1)
def get_collection_router() -> SemanticRouter:
    """Same encoder as retrieval and as the analytical router: one MiniLM for the process."""
    routes = [Route(name=name, utterances=phrases) for name, phrases in sorted(utterances().items())]
    router = SemanticRouter(encoder=HuggingFaceEncoder(name=DENSE_MODEL), routes=routes, auto_sync="local")
    for name, threshold in FITTED_THRESHOLDS.items():
        router.set_threshold(threshold=threshold, route_name=name)
    return router


def classify(question: str) -> str | None:
    """Best-matching collection, or None when the question resembles no part of the corpus."""
    return get_collection_router()(question).name


def refit() -> dict[str, float]:
    """Re-run the threshold search on TRAIN. Run by hand after re-ingesting or changing
    TRAIN, then paste the numbers into FITTED_THRESHOLDS; fit() is a random search."""
    router = get_collection_router()
    questions, labels = zip(*TRAIN)
    router.fit(X=list(questions), y=list(labels))
    return {name: round(float(value), 2) for name, value in router.get_thresholds().items()}
