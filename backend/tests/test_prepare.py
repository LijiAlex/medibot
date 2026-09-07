import pytest
from langchain_core.documents import Document

from medibot.config import COLLECTION_ROLES, EMBED_MAX_TOKENS
from medibot.ingestion.chunker import chunk_document, get_chunker
from medibot.ingestion.loader import load_sources
from medibot.ingestion.parser import parse
from medibot.ingestion.prepare import REQUIRED_METADATA, prepare

SOURCES = {d.source_document: d for d in load_sources()}


def _prepared(name: str) -> list[Document]:
    src = SOURCES[name]
    doc = parse(src)
    return prepare(chunk_document(doc), src, doc)


@pytest.fixture(scope="module")
def faq():
    return _prepared("general_faqs.pdf")


@pytest.fixture(scope="module")
def formulary():
    return _prepared("drug_formulary.pdf")


@pytest.fixture(scope="module")
def guide():
    return _prepared("claim_submission_guide.md")


def test_returns_langchain_documents_with_full_metadata(faq):
    assert faq and all(isinstance(d, Document) for d in faq)
    for d in faq:
        assert REQUIRED_METADATA <= d.metadata.keys(), d.metadata
        assert d.metadata["source_document"] == "general_faqs.pdf"
        assert d.metadata["collection"] == "general"
        assert d.metadata["access_roles"] == COLLECTION_ROLES["general"]


def test_breadcrumb_has_title_and_nested_pdf_headings(faq, formulary):
    pf = next(d for d in faq if "12% employee" in d.page_content)
    assert pf.page_content.startswith("General Staff FAQs > Payroll & Benefits > Q3. What is the Provident Fund")
    assert pf.metadata["section_title"].startswith("Q3.")

    anti = next(d for d in formulary if "Meropenem" in d.page_content)
    assert anti.page_content.startswith("Approved Drug Formulary > 1. Antimicrobials\n")
    assert anti.metadata["section_title"] == "1. Antimicrobials"


def test_breadcrumb_does_not_duplicate_title_when_docling_already_has_it(guide):
    scope = next(d for d in guide if d.metadata["section_title"] == "Purpose & Scope")
    assert scope.page_content.startswith("Claim Submission & Escalation Guide > Purpose & Scope\n")
    assert scope.page_content.count("Claim Submission & Escalation Guide") == 1


def test_chunk_type_rules(faq, formulary, guide):
    assert {d.metadata["chunk_type"] for d in faq} == {"text"}
    assert any(d.metadata["chunk_type"] == "table" for d in formulary)
    fence = next(d for d in guide if "Admission ──► Eligibility check" in d.page_content)
    assert fence.metadata["chunk_type"] == "code"
    inline = next(d for d in guide if d.metadata["section_title"] == "Purpose & Scope")
    assert inline.metadata["chunk_type"] == "text"  # inline `code` spans must not mislabel


def test_point_ids_are_deterministic_and_unique(faq):
    again = _prepared("general_faqs.pdf")
    ids = [d.metadata["point_id"] for d in faq]
    assert ids == [d.metadata["point_id"] for d in again]
    assert len(set(ids)) == len(ids)


def test_final_embedded_text_fits_model_limit(faq, formulary, guide):
    tok = get_chunker().tokenizer
    for d in faq + formulary + guide:
        n = tok.count_tokens(d.page_content)
        assert n <= EMBED_MAX_TOKENS, (n, d.metadata["source_document"], d.metadata["section_title"])
