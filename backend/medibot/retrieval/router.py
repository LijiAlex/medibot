"""Semantic router: the first question /chat asks, before any role check.

    route(question)        -> RouteChoice   name ("analytical" | "documents" | None) + similarity score
    is_analytical(question) -> bool          True only when the analytical route wins

Class reference (Session 6): semantic-router package, Route + HuggingFaceEncoder +
SemanticRouter(auto_sync="local"). Encoder here is the same MiniLM used for dense
retrieval, so no second model is downloaded.

Anything that is not clearly analytical goes to the hybrid branch: it is RBAC-filtered
and runs no SQL, so it is the safe default. The role gate lives in sql_rag.answer, after
this decision, exactly as the spec's flowchart orders it.
"""

from __future__ import annotations

import re
from functools import lru_cache

from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.routers import SemanticRouter
from semantic_router.schema import RouteChoice

from medibot.config import DENSE_MODEL

ANALYTICAL = "analytical"
DOCUMENTS = "documents"

# Utterances are anchored to what the two tables hold (claims, insurers, departments,
# maintenance tickets, campuses, statuses, amounts), not to "how many". A dosage question
# with a number in it must still be a documents question.
ANALYTICAL_UTTERANCES = [
    "How many claims were escalated last month?",
    "Which insurer has the highest number of approved claims?",
    "What is the average claimed amount per department?",
    "How many maintenance tickets are open right now?",
    "Which equipment category has the most open maintenance tickets?",
    "How many tickets were raised at each campus in 2024?",
    "Total approved amount for cashless claims",
    "How many claims are pending with each insurer?",
    "Count rejected claims by department",
    "Average number of days to resolve a maintenance ticket",
    "How many tickets are escalated per campus?",
    "Which department submitted the most claims?",
    "Number of claims by status",
    "Sum of claimed amount for neurology in March 2024",
]

DOCUMENTS_UTTERANCES = [
    "What is the standard dose of amoxicillin?",
    "Which drug tier is ceftriaxone in the formulary?",
    "Which IV cannula gauge for a neonate?",
    "What does fault code F-02 mean on the ventilator?",
    "How do I calibrate the infusion pump?",
    "What is the billing code for a chest X-ray?",
    "How do I submit a reimbursement claim to the insurer?",
    "What documents are needed for cashless claim pre-authorisation?",
    # Billing has many document topics and the analytical set owns every claim phrasing,
    # so "claim" alone used to pull a document lookup across. These four are section
    # titles that exist in the corpus, not phrasings reverse-engineered from a failure.
    "What do the common rejection codes mean?",
    "Which diagnosis codes does the hospital use for billing?",
    "What is the room rent sub-limit for a private ward?",
    "Which exclusions apply to a policy?",
    "What is the hand hygiene procedure before a central line insertion?",
    "Treatment protocol for community-acquired pneumonia",
    "How many days of annual leave am I entitled to?",
    "What is the dress code in the staff handbook?",
    "Who do I report a code of conduct violation to?",
    "Steps for isolating a patient with MRSA",
]


# Labelled questions used to FIT the route thresholds (class reference did the same).
# Distinct from tests/test_router.py, which is the held-out acceptance set.
TRAIN: list[tuple[str, str | None]] = [
    ("How many claims did each insurer reject in 2024?", ANALYTICAL),
    ("Which campus raised the most maintenance tickets?", ANALYTICAL),
    ("Average approved amount for orthopaedics claims", ANALYTICAL),
    ("How many tickets are in progress at Secunderabad?", ANALYTICAL),
    ("Total claimed amount by claim type", ANALYTICAL),
    ("How many claims were submitted in June 2024?", ANALYTICAL),
    ("Which issue type is most common for infusion pumps?", ANALYTICAL),
    ("Number of escalated tickets in the laboratory category", ANALYTICAL),
    ("What is the paediatric dose of paracetamol?", DOCUMENTS),
    ("How often should the ventilator filter be replaced?", DOCUMENTS),
    ("What is the ICD code for acute kidney injury?", DOCUMENTS),
    ("How do I raise a pre-authorisation request?", DOCUMENTS),
    ("PPE sequence for entering an isolation room", DOCUMENTS),
    ("How much maternity leave do staff get?", DOCUMENTS),
    ("What is the escalation path for a harassment complaint?", DOCUMENTS),
    ("First-line antibiotic for urinary tract infection", DOCUMENTS),
    ("What is the weather today?", None),
    ("Tell me a joke", None),
    ("Who won the cricket match?", None),
]

# Output of refit() on 2026-09-09 (MiniLM, top_k=5, mean): train accuracy 0.42 -> 0.95.
# Pinned so every process routes the same way; fit() is a random search and would drift.
# Default 0.5 missed "tickets raised at the Pune campus in November 2024" (top-5 mean 0.45).
FITTED_THRESHOLDS = {ANALYTICAL: 0.40, DOCUMENTS: 0.20}


_TRAILING = re.compile(r"[?\s]+$")


def _canonical(text: str) -> str:
    """One trailing question mark, always.

    Measured 2026-09-11: "How many tickets were raised for sensor failures?" scored 0.433
    against the 0.40 threshold and routed to SQL, while the same words without the mark
    scored 0.349 and fell to documents. A technician typing without punctuation got a
    different branch from one who typed with it. Punctuation is not intent, so both the
    utterances and the incoming question are normalised before anything is embedded.
    Tested three ways over every labelled question asked in both forms: leaving it alone
    scored 31/32, stripping the mark 30/32, forcing it 32/32.
    """
    return _TRAILING.sub("", text.strip()) + "?"


@lru_cache(maxsize=1)
def get_router() -> SemanticRouter:
    encoder = HuggingFaceEncoder(name=DENSE_MODEL)
    routes = [
        Route(name=ANALYTICAL, utterances=[_canonical(u) for u in ANALYTICAL_UTTERANCES]),
        Route(name=DOCUMENTS, utterances=[_canonical(u) for u in DOCUMENTS_UTTERANCES]),
    ]
    router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")
    for name, threshold in FITTED_THRESHOLDS.items():
        router.set_threshold(threshold=threshold, route_name=name)
    return router


def refit() -> dict[str, float]:
    """Re-run the threshold search on TRAIN and return the result. Run by hand after
    changing utterances or TRAIN; paste the numbers into FITTED_THRESHOLDS."""
    router = get_router()
    questions, labels = zip(*TRAIN)
    router.fit(X=[_canonical(q) for q in questions], y=list(labels))
    return {k: round(float(v), 3) for k, v in router.get_thresholds().items()}


def route(question: str) -> RouteChoice:
    """Nearest route by cosine over the utterances (top_k=5, mean), or name=None below threshold."""
    return get_router()(_canonical(question))


def is_analytical(question: str) -> bool:
    return route(question).name == ANALYTICAL
