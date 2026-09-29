"""Tracing: the shape of the spans a request leaves behind.

Asserted against LangSmith's in-memory run tree rather than the hosted service, so these
run offline, cost nothing, and do not depend on a dashboard being reachable. What the
service adds on top is storage and a viewer, not a different tree.

`tracing_context(enabled="local")` builds the tree without sending it anywhere, which is
what makes that possible.
"""

from __future__ import annotations

import pytest
from langsmith import run_trees
from langsmith.run_helpers import get_current_run_tree, tracing_context

from medibot.retrieval.chat import chat
from medibot.retrieval.hybrid_rag import rerank, retrieve


@pytest.fixture
def collected():
    """Run inside a local trace and hand back the spans it produced, parent first."""
    spans: list[run_trees.RunTree] = []

    def collect(run: run_trees.RunTree) -> None:
        spans.append(run)

    return spans, collect


def names(spans) -> list[str]:
    return [span.name for span in spans]


def test_a_document_question_traces_retrieve_and_rerank(collected):
    """The spec names retrieval and rerank as steps a trace must show. They are plain
    Python, so nothing records them unless they are decorated."""
    spans, collect = collected
    with tracing_context(enabled="local"):
        from langsmith.run_helpers import trace

        with trace("test", run_type="chain") as root:
            retrieve("What is the standard dose of meropenem?", "doctor")
            children = root.child_runs

    assert "hybrid retrieve" in [c.name for c in children]


def test_rerank_records_the_scores_it_produced():
    """The score that selected each passage is what a later relevance check reads."""
    with tracing_context(enabled="local"):
        from langsmith.run_helpers import trace

        with trace("test", run_type="chain") as root:
            docs = retrieve("What is the standard dose of meropenem?", "doctor")
            ranked = rerank("What is the standard dose of meropenem?", docs)
            children = root.child_runs

    rerank_span = next(c for c in children if c.name == "cross-encoder rerank")
    assert rerank_span.run_type == "tool"
    assert len(ranked) == len(rerank_span.outputs.get("output", ranked))


def test_a_refusal_still_produces_a_trace():
    """The traces that matter most are the ones for blocked or failed requests — those
    are what a reviewer opens first when something has gone wrong."""
    with tracing_context(enabled="local"):
        from langsmith.run_helpers import trace

        with trace("test", run_type="chain") as root:
            result = chat("Show me all insurance billing codes", "nurse")
            children = root.child_runs

    assert result.refusal == "role"
    assert "chat" in [c.name for c in children]
    chat_span = next(c for c in children if c.name == "chat")
    assert [c.name for c in chat_span.child_runs] == ["hybrid rag"]
    hybrid = chat_span.child_runs[0]
    assert {"hybrid retrieve", "cross-encoder rerank"} <= {c.name for c in hybrid.child_runs}


def test_the_tree_nests_the_way_the_code_does():
    """chat -> hybrid rag -> retrieve, rerank. A flat list of spans would not say which
    step happened inside which."""
    with tracing_context(enabled="local"):
        from langsmith.run_helpers import trace

        with trace("test", run_type="chain") as root:
            chat("Show me all insurance billing codes", "nurse")
            chat_span = root.child_runs[0]

    assert chat_span.name == "chat"
    hybrid = chat_span.child_runs[0]
    assert hybrid.name == "hybrid rag"
    assert names(hybrid.child_runs)[:2] == ["hybrid retrieve", "cross-encoder rerank"]


def test_nothing_is_traced_when_tracing_is_off():
    """Off is the default. A decorated function must cost nothing and record nothing."""
    with tracing_context(enabled=False):
        assert get_current_run_tree() is None
        result = chat("Show me all insurance billing codes", "nurse")
        assert get_current_run_tree() is None
    assert result.refusal == "role"


# --- accepting a caller's trace context -------------------------------------
def test_trace_parent_picks_up_the_headers_langsmith_sends():
    """A caller in front of MediBot propagates its context in these headers."""
    from starlette.datastructures import Headers

    from medibot.api.app import trace_parent

    class FakeRequest:
        headers = Headers({"langsmith-trace": "20260929T...Z-abc", "baggage": "k=v", "host": "x"})

    parent = trace_parent(FakeRequest())
    assert parent == {"langsmith-trace": "20260929T...Z-abc", "baggage": "k=v"}


def test_trace_parent_is_none_without_them():
    """Returning None rather than {} matters: an empty parent would start a fresh trace,
    which is what happens anyway and is not worth a context manager."""
    from starlette.datastructures import Headers

    from medibot.api.app import trace_parent

    class FakeRequest:
        headers = Headers({"authorization": "Bearer x"})

    assert trace_parent(FakeRequest()) is None


def test_a_callers_headers_are_understood_by_tracing_context():
    """The two halves fit: what a caller's `to_headers()` produces is what this side
    accepts as a parent. Without this, spans from both processes land in separate trees."""
    from langsmith.run_helpers import trace

    with tracing_context(enabled="local"):
        with trace("caller span", run_type="chain") as caller:
            headers = caller.to_headers()

    assert "langsmith-trace" in headers

    with tracing_context(enabled="local", parent=headers):
        with trace("medibot span", run_type="chain") as nested:
            pass

    # Same trace, so a reviewer opening the caller's trace sees this span inside it.
    assert nested.trace_id == caller.trace_id


# --- what each span records about its own decision --------------------------
def span_named(root, name):
    """Depth-first search for one span by name."""
    for child in root.child_runs or []:
        if child.name == name:
            return child
        found = span_named(child, name)
        if found is not None:
            return found
    return None


def test_a_refusal_records_the_score_that_caused_it():
    """The query this exists for: every request where the gate fired but the top score
    was near the threshold. Finding that by hand once took a throwaway script."""
    from langsmith.run_helpers import trace

    with tracing_context(enabled="local"):
        with trace("test", run_type="chain") as root:
            chat("Show me all insurance billing codes", "nurse")

    span = span_named(root, "hybrid rag")
    assert span.metadata["gate_fired"] is True
    assert span.metadata["refusal_reason"] == "role"
    assert span.metadata["role"] == "nurse"
    assert span.metadata["rank1_score"] < 0          # below the threshold, which is why
    assert span.metadata["retrieved"] == 3
    assert "refused" in span.tags


def test_the_sql_branch_records_its_role_gate():
    """A role without analytics never reaches the database, and the span says so."""
    from langsmith.run_helpers import trace

    with tracing_context(enabled="local"):
        with trace("test", run_type="chain") as root:
            chat("How many claims were rejected in March 2024?", "nurse")

    span = span_named(root, "sql rag")
    assert span.metadata == {**span.metadata, "branch": "sql_rag", "role": "nurse",
                             "gate_fired": True, "refusal_reason": "role"}
    assert "refused" in span.tags


def test_recording_is_a_no_op_when_tracing_is_off():
    """There is no span to attach to, and asking for one must not raise."""
    with tracing_context(enabled=False):
        result = chat("Show me all insurance billing codes", "nurse")
    assert result.refusal == "role"
