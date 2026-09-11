"""Semantic router: is this an analytical (SQL) question or a document question?

Labelled questions below are the acceptance set. Every run prints the route and its
similarity score so margins are visible, the same way reranker scores were.
"""

import pytest

from medibot.retrieval.router import ANALYTICAL, DOCUMENTS, is_analytical, route

# (question, expected analytical?)  Traps are marked.
LABELLED = [
    # analytical: counts / aggregates over claims and maintenance_tickets
    ("How many claims were escalated in March 2024?", True),
    ("Which insurer has the most approved claims?", True),
    ("What is the average claimed amount for cardiology claims?", True),
    ("Which equipment category has the most open maintenance tickets?", True),
    ("How many tickets were raised at the Pune campus in November 2024?", True),
    ("Count of rejected claims by department", True),
    ("How many maintenance tickets are still open?", True),
    # documents: doses, procedures, codes, policies, manuals
    ("What is the standard dose of meropenem?", False),
    ("Which IV cannula size for a paediatric patient under 5 kg?", False),
    ("What does fault code F-05 mean on the infusion pump?", False),
    ("What is the billing code for an MRI scan?", False),
    ("How do I submit a cashless claim?", False),
    ("What is the hand hygiene protocol before entering the ICU?", False),
    ("How many days of casual leave do staff get?", False),          # trap: "how many" but policy text
    ("Meropenem 1 g every 8 hours, which formulary tier is that?", False),  # trap: numbers, still a doc
    ("Steps to escalate a rejected claim with the insurer", False),  # trap: claim words, but a procedure
]


@pytest.mark.parametrize("question,expected", LABELLED, ids=[q[:40] for q, _ in LABELLED])
def test_labelled_questions_route_correctly(question, expected):
    choice = route(question)
    score = f"{choice.similarity_score:.3f}" if choice.similarity_score is not None else "  -  "
    print(f"\n{choice.name!s:>10}  {score}  {question}")
    assert is_analytical(question) is expected, f"{question!r} -> {choice.name} ({score})"


def test_route_names_are_the_two_we_defined():
    assert {ANALYTICAL, DOCUMENTS} == {"analytical", "documents"}


def test_out_of_scope_is_not_analytical():
    # Nothing matches → semantic-router returns name None → hybrid branch (RBAC-filtered, no SQL).
    assert is_analytical("hello there") is False


def test_pinned_thresholds_still_fit_the_train_set():
    from medibot.retrieval.router import FITTED_THRESHOLDS, TRAIN, get_router
    router = get_router()
    assert router.get_thresholds() == FITTED_THRESHOLDS
    questions, labels = zip(*TRAIN)
    accuracy = router.evaluate(X=list(questions), y=list(labels))
    print(f"\ntrain-set accuracy at pinned thresholds: {accuracy:.3f}")
    assert accuracy >= 0.9


@pytest.mark.parametrize("question,expected", LABELLED, ids=[q[:36] for q, _ in LABELLED])
def test_a_missing_question_mark_does_not_change_the_branch(question, expected):
    """Measured 2026-09-11: without normalisation, dropping the mark moved
    "How many tickets were raised for sensor failures" from SQL to documents."""
    unpunctuated = question.rstrip("? ")
    assert is_analytical(unpunctuated) is expected, f"{unpunctuated!r} routed differently"
