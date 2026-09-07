import re

import pytest
from docling_core.types.doc import DocItemLabel

from medibot.config import EMBED_MAX_TOKENS
from medibot.ingestion.chunker import chunk_document, get_chunker
from medibot.ingestion.loader import load_sources
from medibot.ingestion.parser import parse

SOURCES = {d.source_document: d for d in load_sources()}
Q_LINE = re.compile(r"\bQ\d+\.")


@pytest.fixture(scope="module")
def faq_chunks():
    return chunk_document(parse(SOURCES["general_faqs.pdf"]))


@pytest.fixture(scope="module")
def formulary_chunks():
    return chunk_document(parse(SOURCES["drug_formulary.pdf"]))


def test_faq_gives_one_question_per_chunk(faq_chunks):
    chunker = get_chunker()
    for c in faq_chunks:
        full = chunker.contextualize(c)
        assert len(Q_LINE.findall(full)) <= 1, full[:200]


def test_faq_answer_carries_its_question_as_heading(faq_chunks):
    pf = [c for c in faq_chunks if c.text.startswith("12% employee")]
    assert len(pf) == 1
    assert any("Provident Fund" in h for h in (pf[0].meta.headings or [])), pf[0].meta.headings


def test_page_furniture_is_excluded(faq_chunks, formulary_chunks):
    for c in faq_chunks + formulary_chunks:
        assert "CONFIDENTIAL" not in c.text
        assert "Page " not in c.text


def test_every_chunk_fits_embedding_limit(faq_chunks, formulary_chunks):
    chunker = get_chunker()
    for c in faq_chunks + formulary_chunks:
        n = chunker.tokenizer.count_tokens(chunker.contextualize(c))
        assert n <= EMBED_MAX_TOKENS, (n, c.meta.headings)


def test_formulary_tables_sit_under_their_heading(formulary_chunks):
    anti = [
        c for c in formulary_chunks
        if any("Antimicrobials" in h for h in (c.meta.headings or []))
        and any(it.label == DocItemLabel.TABLE for it in c.meta.doc_items)
    ]
    assert len(anti) >= 1
    assert any("Meropenem" in c.text for c in anti)
