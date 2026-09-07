import math

from langchain_qdrant import FastEmbedSparse
from langchain_huggingface import HuggingFaceEmbeddings

from medibot.config import DENSE_DIM, EMBED_MAX_TOKENS
from medibot.ingestion.embeddings import get_dense, get_sparse


def test_factories_return_cached_singletons():
    assert get_dense() is get_dense()
    assert get_sparse() is get_sparse()
    assert isinstance(get_dense(), HuggingFaceEmbeddings)
    assert isinstance(get_sparse(), FastEmbedSparse)


def test_dense_vector_is_384d_and_unit_length():
    vec = get_dense().embed_query("Meropenem 1 g Q8H IV")
    assert len(vec) == DENSE_DIM
    assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, abs_tol=1e-3)


def test_dense_model_limit_matches_config():
    # The chunker's token budget rests on this number. If the model changes, fail loudly.
    assert get_dense()._client.max_seq_length == EMBED_MAX_TOKENS


def test_dense_is_deterministic_and_semantic():
    d = get_dense()
    a = d.embed_query("dressing change every 72 hours")
    b = d.embed_query("dressing change every 72 hours")
    c = d.embed_query("how often should the CVC dressing be replaced")
    z = d.embed_query("salary is credited on the last working day")
    cos = lambda u, v: sum(x * y for x, y in zip(u, v))
    assert a == b
    assert cos(a, c) > cos(a, z)


def test_sparse_vector_is_keyword_based():
    s = get_sparse()
    v = s.embed_query("APN settings for DriveFlow IP-200")
    assert len(v.indices) == len(v.values) > 0
    # same tokens -> same sparse vector; different keywords -> different indices
    assert s.embed_query("APN settings for DriveFlow IP-200").indices == v.indices
    assert set(v.indices) != set(s.embed_query("provident fund contribution").indices)


def test_embed_documents_batches():
    texts = ["one", "two", "three"]
    assert len(get_dense().embed_documents(texts)) == 3
    assert len(get_sparse().embed_documents(texts)) == 3
