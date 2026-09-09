"""One question in, one RagResult out. The fork the spec's /chat flowchart draws.

    chat(question, role) -> RagResult

Order matters and follows the spec: classify FIRST, gate SECOND.

    is_analytical(question)?
        yes -> sql_rag.answer(question, role)   role gate lives inside; a role without
                                                analytics gets a refusal and no SQL runs
        no  -> hybrid_rag.answer(question, role)  RBAC filter applied inside the Qdrant query

The router never sees the role, so a question is classified the same way for everyone;
only what happens next depends on who is asking. Anything the router cannot place
(name=None) falls to hybrid, which is the safe default: RBAC-filtered and runs no SQL.
"""

from __future__ import annotations

from medibot.retrieval.hybrid_rag import RagResult
from medibot.retrieval.hybrid_rag import answer as hybrid_answer
from medibot.retrieval.router import is_analytical
from medibot.retrieval.sql_rag import answer as sql_answer


def chat(question: str, role: str) -> RagResult:
    if is_analytical(question):
        return sql_answer(question, role)
    return hybrid_answer(question, role)
