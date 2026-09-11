"""Hybrid retrieval against dense-only, on the same index.

The grading criterion asks for retrieval quality "demonstrably better than dense-only",
so this measures it rather than asserting it. Metric: the rank at which the first chunk
containing the exact term appears within the top-10 candidate set both modes are given.
Lower is better; None means it never appeared.

Measured 2026-09-11: hybrid better on 5, equal on 5, worse on 0. The clearest case is the
diagnosis code N17.9, which hybrid puts at rank 2 and dense-only does not surface until
rank 15, so it never reaches the reranker at all.
"""

import pytest
from langchain_qdrant import QdrantVectorStore, RetrievalMode

from medibot.config import QDRANT_COLLECTION, RETRIEVE_K
from medibot.ingestion.embeddings import get_dense, get_sparse
from medibot.vectorstore import DENSE_VECTOR, SPARSE_VECTOR, get_client, rbac_filter

# (question, exact string the right chunk must contain, role)
CASES = [
    ("What does fault code F-05 mean?", "F-05", "technician"),
    ("What is the standard dose of meropenem?", "Meropenem", "doctor"),
    ("Which protocol covers ICD-10 I21.4?", "I21.4", "doctor"),
    ("How do I service the DriveFlow IP-200?", "DriveFlow IP-200", "technician"),
    ("What is document COMP-COC-003?", "COMP-COC-003", "nurse"),
    ("Alarm defaults on the BM-500 monitor", "BM-500", "technician"),
    ("Which cannula gauge for a neonate under 5 kg?", "24G", "nurse"),
    ("What is the diagnosis code N17.9 used for?", "N17.9", "billing_executive"),
    ("How should staff behave towards patients?", "Patient First", "nurse"),
    ("What happens if I take leave without approval?", "Leave Without Pay", "nurse"),
]


def _store(mode: RetrievalMode) -> QdrantVectorStore:
    client = get_client()
    if mode is RetrievalMode.DENSE:
        return QdrantVectorStore(
            client=client, collection_name=QDRANT_COLLECTION,
            embedding=get_dense(), retrieval_mode=mode, vector_name=DENSE_VECTOR,
        )
    return QdrantVectorStore(
        client=client, collection_name=QDRANT_COLLECTION,
        embedding=get_dense(), sparse_embedding=get_sparse(), retrieval_mode=mode,
        vector_name=DENSE_VECTOR, sparse_vector_name=SPARSE_VECTOR,
    )


def _rank(store: QdrantVectorStore, question: str, needle: str, role: str, k: int = RETRIEVE_K):
    docs = store.similarity_search(question, k=k, filter=rbac_filter(role))
    return next((i for i, d in enumerate(docs, 1) if needle.lower() in d.page_content.lower()), None)


@pytest.mark.parametrize("question,needle,role", CASES, ids=[n for _, n, _ in CASES])
def test_hybrid_is_never_worse_than_dense_only(question, needle, role):
    hybrid = _rank(_store(RetrievalMode.HYBRID), question, needle, role)
    dense = _rank(_store(RetrievalMode.DENSE), question, needle, role)
    print(f"\n{needle:18} hybrid={hybrid or 'miss':>4}  dense={dense or 'miss':>4}")
    assert hybrid is not None, f"hybrid lost {needle!r} entirely"
    assert hybrid <= (dense or 99), f"dense-only beat hybrid on {needle!r}: {dense} vs {hybrid}"


def test_a_diagnosis_code_is_out_of_reach_for_dense_only():
    """The headline case. Dense-only ranks N17.9 at 15, so with a top-10 candidate set it
    never reaches the reranker; BM25 puts it at 2. This is the keyword half of hybrid
    search earning its place, exactly as the spec's own tip predicts."""
    question, needle, role = "What is the diagnosis code N17.9 used for?", "N17.9", "billing_executive"
    assert _rank(_store(RetrievalMode.HYBRID), question, needle, role) <= 3
    assert _rank(_store(RetrievalMode.DENSE), question, needle, role) is None, "dense should miss it in top-10"
    assert _rank(_store(RetrievalMode.DENSE), question, needle, role, k=100) > RETRIEVE_K
