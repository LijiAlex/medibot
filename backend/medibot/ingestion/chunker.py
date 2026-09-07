"""Step 3 of ingestion: DoclingDocument -> list[DocChunk].

HybridChunker does two passes:
  1. hierarchical: one candidate chunk per leaf item (paragraph, list, table),
     tagged with the heading path it sits under. Headings themselves are never
     chunk bodies; they only update the "current heading" at their level.
  2. token-aware: split anything over max_tokens (a 16-row table becomes
     several row windows), and merge undersized siblings that share the same
     heading path (merge_peers), so a 9-step procedure stays one chunk.

The tokenizer MUST be the dense embedding model's own tokenizer. MiniLM silently
truncates at 256 tokens; a chunk measured with a different tokenizer can exceed
that and lose its tail at embedding time.

Output chunks are raw Docling chunks: body text + heading path + source items.
Breadcrumb text, metadata and ids are built in the next step (prepare).
"""

from __future__ import annotations

from functools import lru_cache

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import DocChunk
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import DoclingDocument
from transformers import AutoTokenizer

from medibot.config import CHUNK_MAX_TOKENS, DENSE_MODEL


@lru_cache(maxsize=4)
def get_chunker(max_tokens: int = CHUNK_MAX_TOKENS) -> HybridChunker:
    """Build (once per max_tokens value) the HybridChunker used for every document.

    max_tokens is the cap on chunk *body* text. It is deliberately smaller than
    the embedding model's limit (224 vs 256): the heading breadcrumb is added on
    top of the body later, and HybridChunker measures oversized-table splits on
    body text alone. The headroom keeps the final embedded text under 256.

    Cached because loading the tokenizer takes time and the chunker is stateless.
    """
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(DENSE_MODEL),
        max_tokens=max_tokens,
    )
    return HybridChunker(tokenizer=tokenizer, merge_peers=True)


def chunk_document(doc: DoclingDocument, max_tokens: int = CHUNK_MAX_TOKENS) -> list[DocChunk]:
    """Chunk one parsed document.

    Returns chunks in reading order; the list index becomes chunk_index in the
    prepare step, which is what makes point ids deterministic.

    Each DocChunk carries:
        .text            body only (no headings), <= max_tokens
        .meta.headings   heading path, e.g. ["ICU Manual", "SOP 1 - CVC Care", "Frequency"]
        .meta.doc_items  the source items it was built from (used for chunk_type)
    """
    return list(get_chunker(max_tokens).chunk(dl_doc=doc))
