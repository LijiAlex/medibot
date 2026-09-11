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
    CLASSIFY_MARGIN,
    LLM_MODEL,
    RELEVANCE_THRESHOLD,
    RERANK_MODEL,
    RERANK_TOP_N,
    RETRIEVE_K,
    ROLE_COLLECTIONS,
)
from medibot.retrieval.collection_router import classify_against
from medibot.retrieval.phrasing import english_list, refusal
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
    # Why there is no answer, when there is none. An empty sources list alone cannot tell
    # a permission decision from an empty search, and the label on screen must not claim
    # "nothing matched" about a question that was never searched for.
    #   "role"      the role may not read that collection, or may not query the records
    #   "not_found" searched, nothing relevant
    #   "no_query"  the model produced nothing that cleaned to a SELECT
    refusal: str | None = None


SYSTEM_PROMPT = """You are MediBot, an internal assistant for hospital staff.
Answer the question using ONLY the numbered context passages below. Be specific:
quote doses, codes, sizes and steps exactly as written. Cite the passages you used
as [1], [2], [3]. If the context does not contain the answer, say so and do not guess.

The passages are ordered by relevance, [1] being the most relevant. Use every passage
that bears on the question, not only the first that matches. Where they set out
different cases, conditions or staff groups, give all of them: a reader who asks what a
rule is must not be handed one case as though it were the whole rule.

If the question names a document, treat that as a hint about the topic, never as a
filter. Answer from whichever passages actually state the rule, whatever file they came
from, and say where each part came from.

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


def _refusal(question: str, role: str) -> tuple[str, str]:
    """Why nothing came back, in the words the spec asks for (Component 6).

    Two different situations reach here, and conflating them would mean telling a doctor
    they lack access to the clinical collection they can read:

      the question belongs to a collection this role may not read  -> say so, name it
      the question belongs to one they may read, or to none at all -> say we found nothing

    The classifier only ever runs here, on a question retrieval has already failed, so a
    misclassification changes the wording of a refusal and never causes one.
    """
    permitted = ROLE_COLLECTIONS[role]
    collection, margin = classify_against(question, permitted)
    if collection is not None and collection not in permitted and margin >= CLASSIFY_MARGIN:
        return refusal(role, f"{collection} documents", inferred=True), "role"
    return (
        "I couldn't find anything about that in the documents you can access "
        f"({english_list(ROLE_COLLECTIONS[role])})."
    ), "not_found"


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
        text, why = _refusal(question, role)
        logger.info("hybrid_rag refused role=%s why=%s top1=%s question=%r", role, why, top, question)
        return RagResult(answer=text, retrieval_type="hybrid_rag", role=role, refusal=why)
    chain = PROMPT | get_llm() | StrOutputParser()
    text = chain.invoke({"context": _format_context(ranked), "question": question})
    return RagResult(answer=text, sources=_sources(ranked), retrieval_type="hybrid_rag", role=role)
