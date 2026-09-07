import pytest
from docling_core.types.doc import DocItemLabel, DoclingDocument

from medibot.ingestion.chunker import chunk_document
from medibot.ingestion.loader import load_sources
from medibot.ingestion.parser import parse, strip_inline_markdown

SOURCES = {d.source_document: d for d in load_sources()}
FURNITURE = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER, DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}


@pytest.fixture(scope="module")
def formulary() -> DoclingDocument:
    return parse(SOURCES["drug_formulary.pdf"])


@pytest.fixture(scope="module")
def icu() -> DoclingDocument:
    return parse(SOURCES["icu_nursing_procedures.pdf"])


@pytest.fixture(scope="module")
def guide() -> DoclingDocument:
    return parse(SOURCES["claim_submission_guide.md"])


@pytest.fixture(scope="module")
def diagnostics() -> DoclingDocument:
    return parse(SOURCES["diagnostic_reference.pdf"])


def _headers(doc: DoclingDocument) -> dict[str, int]:
    return {t.text.strip(): t.level for t in doc.texts if t.label == DocItemLabel.SECTION_HEADER}


def test_pdf_parses_to_docling_document(formulary):
    assert isinstance(formulary, DoclingDocument)
    assert len(formulary.texts) > 0


def test_pdf_headings_are_recognised_not_flattened(formulary):
    assert any("Antimicrobials" in h for h in _headers(formulary)), list(_headers(formulary))


def test_pdf_tables_are_recognised(formulary):
    assert len(formulary.tables) >= 3


def test_pdf_subheadings_nest_under_their_section(icu):
    h = _headers(icu)
    sop1 = next(k for k in h if k.startswith("SOP 1"))
    assert h[sop1] < h["Frequency"], (h[sop1], h["Frequency"])
    assert h[sop1] < h["Equipment checklist"]


def test_markdown_parses_with_headings_and_code(guide):
    h = _headers(guide)
    assert any("Cashless Claim Process" in k for k in h), list(h)
    assert any(t.label == DocItemLabel.CODE for t in guide.texts)


def test_markdown_inline_formatting_does_not_fragment_list_items(guide):
    chunks = chunk_document(guide)
    covered = {it.self_ref for c in chunks for it in c.meta.doc_items}
    content = [it for it, _ in guide.iterate_items() if it.label not in FURNITURE]
    missing = [it for it in content if it.self_ref not in covered]
    assert not missing, [getattr(m, "text", "")[:40] for m in missing[:5]]
    joined = " ".join(c.text for c in chunks)
    assert "Admission note with provisional diagnosis" in joined


def test_footnotes_are_kept_as_text(diagnostics):
    assert not any(t.label == DocItemLabel.FOOTNOTE for t in diagnostics.texts)
    joined = " ".join(c.text for c in chunk_document(diagnostics))
    assert "Urgent radiology turnaround" in joined


def test_strip_inline_markdown_leaves_fences_alone():
    src = "1. **Admission note** with `billing_codes.pdf`.\n```\nkeep **this** `as is`\n```\n**multi\nline**"
    out = strip_inline_markdown(src)
    assert out.startswith("1. Admission note with billing_codes.pdf.")
    assert "keep **this** `as is`" in out
    assert out.endswith("multi\nline")
