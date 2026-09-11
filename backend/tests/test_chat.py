"""/chat routing: semantic router first, then the role gate. Spec, Component 5 flowchart."""

import os

import pytest

from medibot.config import SQL_RAG_ROLES
from medibot.retrieval import sql_rag
from medibot.retrieval.chat import chat
from medibot.retrieval.hybrid_rag import RagResult

needs_llm = pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")

SPEC_SOURCE_KEYS = {"source_document", "section_title", "collection"}


@needs_llm
def test_analytical_question_from_a_permitted_role_runs_sql():
    res = chat("How many claims were escalated in March 2024?", "billing_executive")
    assert isinstance(res, RagResult)
    assert res.retrieval_type == "sql_rag" and res.role == "billing_executive"
    assert res.sql.upper().startswith("SELECT")
    assert all(set(s) == SPEC_SOURCE_KEYS for s in res.sources)
    assert res.sources[0]["collection"] == "sql"


def test_analytical_question_from_a_denied_role_never_reaches_the_database(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("write_sql must not run for a role without analytics")

    monkeypatch.setattr(sql_rag, "write_sql", boom)
    for role in ("nurse", "doctor", "technician"):
        assert role not in SQL_RAG_ROLES
        res = chat("Which insurer has the most approved claims?", role)
        assert res.retrieval_type == "sql_rag" and res.sources == [] and res.sql is None
        # Wording unified 2026-09-11 with the document refusal: closed, then still open.
        assert "don't have access to the claims and ticket records" in res.answer


@needs_llm
def test_document_question_goes_to_hybrid_with_citations():
    res = chat("What is the standard dose of meropenem?", "doctor")
    assert res.retrieval_type == "hybrid_rag" and res.sql is None
    assert len(res.sources) == 3
    assert all(set(s) == SPEC_SOURCE_KEYS for s in res.sources)
    assert res.sources[0]["source_document"] == "drug_formulary.pdf"


@needs_llm
def test_out_of_scope_question_falls_to_hybrid_not_sql():
    # Router returns name=None. Hybrid is the safe default: RBAC-filtered, no SQL.
    res = chat("hello there", "admin")
    assert res.retrieval_type == "hybrid_rag" and res.sql is None


def test_routing_does_not_depend_on_role(monkeypatch):
    seen = []
    monkeypatch.setattr("medibot.retrieval.chat.is_analytical", lambda q: seen.append(q) or False)
    monkeypatch.setattr("medibot.retrieval.chat.hybrid_answer", lambda q, r: RagResult(answer="", role=r))
    for role in ("nurse", "admin", "billing_executive"):
        chat("How many claims were escalated in March 2024?", role)
    assert seen == ["How many claims were escalated in March 2024?"] * 3
