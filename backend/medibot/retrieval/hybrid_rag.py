"""Hybrid RAG: the document branch of /chat.

    retrieve(question, role) -> 10 Documents   one Qdrant query, dense + sparse fused, RBAC-filtered
    rerank(question, docs)   -> 3 (Document, score)   cross-encoder, question and chunk read together
    answer(question, role)   -> RagResult        LLM over the 3 reranked chunks, with citations

RBAC is not a post-filter here. rbac_filter(role) goes into the Qdrant query, so
chunks the role may not see are never returned to this process at all.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from medibot.config import (
    LLM_MODEL,
    RELEVANCE_THRESHOLD,
    RERANK_MODEL,
    RERANK_TOP_N,
    RETRIEVE_K,
    ROLE_COLLECTIONS,
)
from medibot.retrieval.collection_router import classify
from medibot.vectorstore import get_vectorstore, rbac_filter


logger = logging.getLogger(__name__)


def retrieve(question: str, role: str, k: int = RETRIEVE_K) -> list[Document]:
    """Broad candidate set for one question, restricted to what the role may read.

    One call: Qdrant embeds the question twice (dense for meaning, sparse for exact
    terms like "F-05" or "meropenem"), runs both searches server-side, fuses the two
    rankings with Reciprocal Rank Fusion, and returns the top k. The filter is
    applied inside Qdrant before ranking, not on the returned list.
    """
    return get_vectorstore().similarity_search(question, k=k, filter=rbac_filter(role))


@lru_cache(maxsize=1)
def get_reranker() -> HuggingFaceCrossEncoder:
    """Cross-encoder, loaded once per process (~90 MB, ~2 s)."""
    return HuggingFaceCrossEncoder(model_name=RERANK_MODEL)


def rerank(question: str, docs: list[Document], top_n: int = RERANK_TOP_N) -> list[tuple[Document, float]]:
    """Second, slower scoring pass: keep the top_n chunks most relevant to the question.

    A bi-encoder (MiniLM at retrieval) embedded question and chunk separately and
    compared the two vectors. A cross-encoder reads them together as one input and
    outputs one relevance score, so it can see which words of the question the
    chunk actually answers. Too slow to run over the whole store; fine over 10.

    Returns (doc, score) pairs, highest first, so the API can log the scores.
    """
    if not docs:
        return []
    scores = get_reranker().score([(question, d.page_content) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return [(d, float(s)) for d, s in ranked[:top_n]]


@dataclass
class RagResult:
    """What /chat returns, for every branch (hybrid, SQL, refusal). Spec, Component 5."""

    answer: str
    sources: list[dict] = field(default_factory=list)  # {source_document, section_title, collection}
    retrieval_type: str = "hybrid_rag"                 # "hybrid_rag" | "sql_rag"
    role: str = ""
    sql: str | None = None                             # SQL branch only; spec line 168 is a minimum


SYSTEM_PROMPT = """You are MediBot, an internal assistant for hospital staff.
Answer the question using ONLY the numbered context passages below. Be specific:
quote doses, codes, sizes and steps exactly as written. Cite the passages you used
as [1], [2], [3]. If the context does not contain the answer, say so and do not guess.

Context:
{context}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Groq-hosted open model. temperature=0: same context, same answer."""
    return ChatGroq(model=LLM_MODEL, temperature=0)


def _format_context(ranked: list[tuple[Document, float]]) -> str:
    """Numbered passages. page_content already starts with the heading breadcrumb, so
    the LLM sees where each passage came from without a separate label."""
    return "\n\n".join(f"[{i}] {doc.page_content}" for i, (doc, _) in enumerate(ranked, start=1))


def _sources(ranked: list[tuple[Document, float]]) -> list[dict]:
    return [
        {k: doc.metadata[k] for k in ("source_document", "section_title", "collection")}
        for doc, _ in ranked
    ]


def _english_list(items: list[str]) -> str:
    """['a', 'b', 'c'] -> 'a, b and c'."""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _describe(role: str) -> str:
    """'billing_executive' -> 'a billing executive'."""
    words = role.replace("_", " ")
    return f"{'an' if words[0] in 'aeiou' else 'a'} {words}"


def _refusal(question: str, role: str) -> str:
    """Why nothing came back, in the words the spec asks for (Component 6).

    Two different situations reach here, and conflating them would mean telling a doctor
    they lack access to the clinical collection they can read:

      the question belongs to a collection this role may not read  -> say so, name it
      the question belongs to one they may read, or to none at all -> say we found nothing

    The classifier only ever runs here, on a question retrieval has already failed, so a
    misclassification changes the wording of a refusal and never causes one.
    """
    permitted = ROLE_COLLECTIONS[role]
    collection = classify(question)
    if collection is not None and collection not in permitted:
        return (
            f"As {_describe(role)}, you don't have access to {collection} documents. "
            f"I can only answer questions from the {_english_list(permitted)} collections."
        )
    return (
        "I couldn't find anything about that in the documents you can access "
        f"({_english_list(permitted)})."
    )


def answer(question: str, role: str) -> RagResult:
    """retrieve -> rerank -> LLM. Only the reranked top_n reaches the prompt (spec, Component 3).

    The gate in between: if the best chunk scores below RELEVANCE_THRESHOLD, nothing
    retrieved answers the question, so we explain why instead of asking the LLM to
    improvise over irrelevant context. That path costs no Groq call and cites nothing,
    because nothing was used.
    """
    ranked = rerank(question, retrieve(question, role))
    if not ranked or ranked[0][1] < RELEVANCE_THRESHOLD:
        top = ranked[0][1] if ranked else None
        logger.info("hybrid_rag refused role=%s top1=%s question=%r", role, top, question)
        return RagResult(answer=_refusal(question, role), retrieval_type="hybrid_rag", role=role)
    chain = PROMPT | get_llm() | StrOutputParser()
    text = chain.invoke({"context": _format_context(ranked), "question": question})
    return RagResult(answer=text, sources=_sources(ranked), retrieval_type="hybrid_rag", role=role)
