"""Step 2 of ingestion: SourceDoc -> DoclingDocument.

Docling reads the file with structural awareness (headings, tables, code blocks,
reading order) and returns a tree we can chunk along. One converter handles PDF
and Markdown; it picks the backend from the file extension.

Three corpus-specific adjustments, each found by inspecting real output:

1. Heading levels (PDF). Docling's layout model flags headings but not their
   rank, so every PDF heading comes out at level 1 and sub-headings overwrite
   their parent in the chunk breadcrumb ("Manual > Frequency" instead of
   "Manual > SOP 1 - CVC Care > Frequency"). Docling's built-in
   HeadingHierarchyOptions infers levels from bookmarks, numbering and font
   style. Our PDFs have no bookmarks; numbering + style is enough.
   generate_parsed_pages=True keeps the text cells the style pass needs.

2. Inline markdown formatting. Docling's markdown backend splits a list item
   like "1. **Admission note** with ... `billing_codes.pdf`." into a list item
   plus loose text fragments, and the chunker drops the fragments (85 of 239
   items in the billing guide). Stripping **bold** and `inline code` markers
   before parsing keeps each list item whole. Fenced code blocks are untouched.

3. Footnotes. The chunker serialises a table as one unit and never emits its
   footnote children; two table footnotes in the diagnostic reference carry
   real rules. They are re-inserted as TEXT siblings right after the table.

Parsing also gets a sanity check: the same file has come back with a different
item set in the same session (a missed heading, once a half-parsed document).
Root cause turned out to be a docling-parse threading race (see _converter);
the check stays as a safety net. Every page must yield content and the document
must have at least one heading, every page in the file must come back, and at
least 80% of the raw text-line words must survive into content items; otherwise
parse once more, then fail loudly.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from io import BytesIO

# onnxruntime 1.29 (pulled in by rapidocr) aborts at interpreter exit in its telemetry
# thread (recursive_mutex lock failed, exit 134). Harmless to results, noisy in logs.
# Must be set before onnxruntime is imported, which Docling does below.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

from docling.backend.docling_parse_backend import ThreadedDoclingParseBackendOptions
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import HeadingHierarchyOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel, DoclingDocument

from medibot.ingestion.loader import SourceDoc

log = logging.getLogger(__name__)

FURNITURE = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}
MIN_CONTENT_RATIO = 0.8  # good parses keep ~85-90% of raw words (rest is header/footer furniture)

_FENCE = re.compile(r"(```.*?```)", re.DOTALL)
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_INLINE_CODE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_LIST_START = re.compile(r"^\s*(\d+\.|[-*+])\s+\S")
_CONTINUATION = re.compile(r"^\s{2,}\S")


@lru_cache(maxsize=1)
def _converter() -> DocumentConverter:
    """Build the one DocumentConverter used for all files (cached: model loading is slow).

    PDF options:
      generate_parsed_pages=True   keep the raw text cells from the PDF parser. Needed
                                   by the font-style heading pass and by our sanity check.
      heading_hierarchy_options    Docling's built-in inference of SectionHeaderItem.level
                                   from bookmarks -> numbering -> font style. Without it
                                   every PDF heading is level 1.
    Markdown needs no options; the converter picks the backend from the file suffix.
    """
    pdf_opts = PdfPipelineOptions(
        generate_parsed_pages=True,
        heading_hierarchy_options=HeadingHierarchyOptions(enabled=True),
    )
    # docling-parse <= 7.16.0 has a race in its font-metrics cache when Docling's
    # default threaded backend decodes pages on 4 native threads: intermittent
    # heap corruption (exit 134/139) and, worse, silent per-page decode failures
    # that look like a normal parse with content missing (docling#4147). Reported
    # fixed in docling-parse 7.17.0 (2026-09-02), but the issue thread and release
    # notes do not confirm it. Single-threaded parsing avoids it; measured cost on
    # a warm 8-page parse was 5.0 s -> 5.1 s. TODO after 2026-09-09 (cooldown):
    # upgrade docling-parse >= 7.17.0, then prove stability (parse a 9-page PDF
    # 6x with the default threaded backend, all pages present every time) BEFORE
    # dropping backend_options.
    backend_opts = ThreadedDoclingParseBackendOptions(parser_threads=1)
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts, backend_options=backend_opts),
        }
    )


def _unwrap_list_items(text: str) -> str:
    """Join indented continuation lines onto the list item they belong to.

    Markdown authors hard-wrap long bullets:
        1. Verify patient eligibility on the insurer portal (policy active,
           waiting period cleared, room-rent category).
    Docling's markdown backend makes the second line a separate text item, which
    splits the sentence and leaves a newline inside it. Joining them first gives
    one list item with the whole sentence.

    Rule: a line indented by 2+ spaces, directly following a list item, and not
    itself a list marker, is appended to the previous line. Anything else passes
    through unchanged.
    """
    out: list[str] = []
    in_list_item = False
    for line in text.split("\n"):
        if in_list_item and _CONTINUATION.match(line) and not _LIST_START.match(line):
            out[-1] = out[-1].rstrip() + " " + line.strip()
            continue
        in_list_item = bool(_LIST_START.match(line))
        out.append(line)
    return "\n".join(out)


def strip_inline_markdown(text: str) -> str:
    """Pre-process markdown text before Docling sees it.

    Outside fenced ``` blocks:
      - **bold**  -> bold          (also when the bold spans a line break)
      - `code`    -> code          (inline spans only; fences are left intact)
      - hard-wrapped list items are joined onto one line

    Why: Docling's markdown backend starts a new item at every formatting boundary,
    so "1. **Admission note** with ... `billing_codes.pdf`." became six items. The
    chunker stitched the text back together but kept the markers and breaks. After
    this pass Docling emits one clean list item, and the embedded text carries no
    markup. Words are never changed; the file on disk is never touched.
    """
    parts = _FENCE.split(text)
    for i, part in enumerate(parts):
        if part.startswith("```"):
            continue
        part = _BOLD.sub(r"\1", part)
        part = _INLINE_CODE.sub(r"\1", part)
        parts[i] = _unwrap_list_items(part)
    return "".join(parts)


def _convert(source: SourceDoc) -> ConversionResult:
    """Run Docling on one file.

    Markdown: read the file, pre-process the text, and hand Docling an in-memory
    DocumentStream (the name keeps the .md suffix so the right backend is chosen).
    PDF: give Docling the path directly; nothing to clean first.
    """
    if source.path.suffix.lower() == ".md":
        cleaned = strip_inline_markdown(source.path.read_text(encoding="utf-8"))
        stream = DocumentStream(name=source.path.name, stream=BytesIO(cleaned.encode("utf-8")))
        return _converter().convert(stream)
    return _converter().convert(str(source.path))


def _hoist_table_footnotes(doc: DoclingDocument) -> None:
    """Post-process: move footnotes out from under their table so the chunker sees them.

    Docling attaches lines printed under a table as FOOTNOTE children of the
    table. The chunker serialises a table as one unit, marks all its children
    visited, and emits only captions, so the footnotes vanish. Relabelling them
    does not help; their position in the tree is the problem.

    Fix: insert each footnote's text as a TEXT item right after the table (in
    order), then delete the originals. They become ordinary paragraphs under the
    same heading.

    Example: diagnostic_reference.pdf, table 6 -> "Urgent radiology turnaround:
    verbal/critical alert within 1 hour" is now retrievable.
    """
    for table in doc.tables:
        notes = [ref.resolve(doc) for ref in table.children]
        notes = [n for n in notes if getattr(n, "label", None) == DocItemLabel.FOOTNOTE]
        if not notes:
            continue
        anchor = table
        for note in notes:
            anchor = doc.insert_text(sibling=anchor, label=DocItemLabel.TEXT, text=note.text, prov=note.prov[0] if note.prov else None, after=True)
        doc.delete_items(node_items=notes)


def _sanity_problems(result: ConversionResult) -> list[str]:
    """Post-process check: return a list of reasons this parse looks incomplete (empty = OK).

    Docling's layout model is occasionally non-deterministic on the same file: a
    missed heading once, and once staff_handbook.pdf came back with 54 items
    instead of 94. Three signals catch that:
      - at least one heading exists
      - every PDF page produced at least one content item
      - at least MIN_CONTENT_RATIO of the raw text-line words (from the PDF parser,
        before the layout model) survive into content items. Table words are
        counted from table cells because TableItem has no .text. Good parses
        score 0.83-0.92; furniture (page headers/footers) accounts for the rest.
    Markdown has no pages, so only the heading check applies to it.
    """
    doc = result.document
    problems: list[str] = []
    if not any(t.label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE) for t in doc.texts):
        problems.append("no headings detected")
    if result.pages:  # PDF only; markdown has no pages
        # Under the docling-parse threading race, lost pages are ABSENT from result.pages,
        # not present-and-empty, and their raw text lines vanish with them, so neither the
        # empty-page test nor the word ratio below can see it. Compare against the file's
        # true page count first.
        expected = getattr(result.input, "page_count", 0) or 0
        if expected and len(result.pages) != expected:
            problems.append(f"only {len(result.pages)} of {expected} pages returned")
        per_page: dict[int, int] = {}
        for item, _ in doc.iterate_items():
            if item.label not in FURNITURE and item.prov:
                per_page[item.prov[0].page_no] = per_page.get(item.prov[0].page_no, 0) + 1
        empty = [p.page_no for p in result.pages if per_page.get(p.page_no, 0) == 0]  # both 1-based
        if empty:
            problems.append(f"pages with no content items: {empty}")
        # Layout model occasionally half-parses a page set: items still exist on every
        # page but a third of the words are gone. Compare against the raw text lines
        # from the PDF parser (available because generate_parsed_pages=True).
        raw_words = sum(
            len(cell.text.split())
            for p in result.pages if p.parsed_page is not None
            for cell in p.parsed_page.textline_cells
        )
        item_words = 0
        for item, _ in doc.iterate_items():
            if item.label in FURNITURE:
                continue
            if item.label == DocItemLabel.TABLE:  # table text lives in cells, not item.text
                item_words += sum(len(c.text.split()) for c in item.data.table_cells)
            else:
                item_words += len((getattr(item, "text", "") or "").split())
        if raw_words and item_words / raw_words < MIN_CONTENT_RATIO:
            problems.append(f"only {item_words}/{raw_words} raw words survived parsing ({item_words / raw_words:.0%})")
    return problems


def parse(source: SourceDoc, retries: int = 1) -> DoclingDocument:
    """Parse one source file into Docling's structured document tree.

    Three phases:
      pre-process   markdown only, in _convert (strip markers, unwrap lists)
      docling       DocumentConverter with heading levels enabled
      post-process  sanity check (retry once on failure, then raise) and
                    footnote hoist

    Returns the DoclingDocument the chunker will walk. Raises RuntimeError if
    the parse still fails the sanity check after `retries` extra attempts, so a
    half-parsed document never reaches the vector store silently.
    """
    for attempt in range(retries + 1):
        result = _convert(source)
        problems = _sanity_problems(result)
        if not problems:
            _hoist_table_footnotes(result.document)
            return result.document
        log.warning("parse of %s attempt %d: %s", source.source_document, attempt + 1, "; ".join(problems))
    raise RuntimeError(f"{source.source_document}: parse failed sanity check after {retries + 1} attempts: {problems}")
