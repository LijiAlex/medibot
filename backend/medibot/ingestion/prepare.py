"""Step 4 of ingestion: list[DocChunk] + SourceDoc -> list[langchain Document].

This is where the assignment's metadata contract is met. For every chunk:

  page_content  breadcrumb + body. The breadcrumb is the heading path with the
                document title forced in front when Docling left it out (numbered
                "1." sections come out at the same level as the title, so the
                title drops off their path).
  metadata      the five required fields (source_document, collection,
                access_roles, section_title, chunk_type) plus chunk_index and a
                deterministic point_id so re-ingestion overwrites instead of
                duplicating.

The same page_content string is what gets embedded (dense and sparse) and what
is stored in Qdrant for the LLM to read. One string, three uses.
"""

from __future__ import annotations

import uuid

from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
from docling_core.types.doc import DocItemLabel, DoclingDocument
from langchain_core.documents import Document

from medibot.ingestion.loader import SourceDoc

# The five fields the assignment requires on every stored chunk.
REQUIRED_METADATA = {"source_document", "collection", "access_roles", "section_title", "chunk_type"}
BREADCRUMB_SEP = " > "


def document_title(doc: DoclingDocument, fallback: str) -> str:
    """Pick the document's title: first TITLE item, else first SECTION_HEADER, else fallback.

    Markdown files have a real TITLE item ("# Claim Submission & Escalation Guide").
    PDFs don't; their big first heading ("Approved Drug Formulary") is the best
    stand-in. The fallback is the filename stem, only reached for a file with no
    headings at all.
    """
    for label in (DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER):
        for t in doc.texts:
            if t.label == label and t.text.strip():
                return t.text.strip()
    return fallback


def chunk_type(chunk: DocChunk) -> str:
    """Classify a chunk from the labels of the items it was built from.

        code   every source item is a code block  -> a real fenced block
        table  at least one source item is a table
        text   everything else

    "All items are code" (not "any item is code") matters: inline `code` spans in
    prose can show up as code items mixed with text, and must not relabel a
    paragraph as code.
    """
    labels = [it.label for it in chunk.meta.doc_items]
    if labels and all(lab == DocItemLabel.CODE for lab in labels):
        return "code"
    if DocItemLabel.TABLE in labels:
        return "table"
    return "text"


def point_id(source_document: str, chunk_index: int) -> str:
    """Deterministic Qdrant point id: same file + same position -> same UUID.

    uuid5 hashes "drug_formulary.pdf:4" into a stable UUID. Re-running ingestion
    upserts over the previous point instead of adding a duplicate. Random UUIDs
    (LangChain's default) would double the collection on every run.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_document}:{chunk_index}"))


def prepare(chunks: list[DocChunk], source: SourceDoc, doc: DoclingDocument) -> list[Document]:
    """Turn raw Docling chunks into LangChain Documents ready for the vector store.

    Per chunk:
      1. headings = the chunk's heading path, title prepended if missing
         -> breadcrumb "Approved Drug Formulary > 1. Antimicrobials"
      2. page_content = breadcrumb + newline + body
      3. section_title = last heading (what the citation card shows)
      4. metadata = 5 required fields + chunk_index + point_id

    Example result for a formulary table chunk:
        page_content: "Approved Drug Formulary > 1. Antimicrobials\\nMeropenem, Route = IV. ..."
        metadata:     {source_document: drug_formulary.pdf, collection: clinical,
                       access_roles: [doctor, admin], section_title: 1. Antimicrobials,
                       chunk_type: table, chunk_index: 4, point_id: 6b98...}
    """
    title = document_title(doc, fallback=source.path.stem)
    out: list[Document] = []
    for idx, chunk in enumerate(chunks):
        headings = [h.strip() for h in (chunk.meta.headings or []) if h and h.strip()]
        if not headings or headings[0] != title:
            headings = [title, *headings]  # restore the title Docling dropped
        breadcrumb = BREADCRUMB_SEP.join(headings)
        section_title = headings[-1]

        out.append(
            Document(
                page_content=f"{breadcrumb}\n{chunk.text}",
                metadata={
                    "source_document": source.source_document,
                    "collection": source.collection,
                    "access_roles": list(source.access_roles),
                    "section_title": section_title,
                    "chunk_type": chunk_type(chunk),
                    "chunk_index": idx,
                    "point_id": point_id(source.source_document, idx),
                },
            )
        )
    return out
