"""Regression tests for embedding-dimension handling (J5 / CH-687).

Every backend must store vectors at full dimension and fail loud on any
dimension mismatch — never silently pad, truncate, or zero-score. Before this
fix the Kuzu backend silently truncated real embeddings to a default of 8 dims,
and `cosine_similarity` returned 0.0 on a length mismatch (a silent wrong
ranking in SQLite / the in-memory graph / episodic recall).
"""

from __future__ import annotations

import pytest

from synap._utils import cosine_similarity
from synap.backends.kuzu import KuzuBackend
from synap.backends.sqlite import SQLiteBackend


def _node(id: str = "n1", embedding=None):
    return {
        "id": id,
        "node_type": "semantic",
        "content": "c",
        "embedding": [0.1, 0.2, 0.3] if embedding is None else embedding,
        "utility_score": 1.0,
        "access_count": 0,
        "created_at": "2026-01-01T00:00:00Z",
        "last_accessed": "2026-01-01T00:00:00Z",
        "metadata": {},
    }


# --- cosine util: fail loud on dimension mismatch ---


def test_cosine_similarity_raises_on_dimension_mismatch():
    with pytest.raises(ValueError):
        cosine_similarity([1.0, 0.0, 0.0], [1.0, 0.0])


def test_cosine_similarity_ok_on_equal_length():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


# --- Kuzu: dim required, no silent truncation, reopen-safe ---


def test_kuzu_embedding_dim_is_required():
    with pytest.raises(TypeError):
        KuzuBackend("unused")  # embedding_dim no longer defaults


def test_kuzu_rejects_wrong_dim_on_write(tmp_path):
    db = KuzuBackend(tmp_path / "g", embedding_dim=3)
    try:
        with pytest.raises(ValueError):
            db.save_node(_node(embedding=[0.1, 0.2, 0.3, 0.4, 0.5]))  # 5 != 3
    finally:
        db.close()


def test_kuzu_rejects_wrong_dim_query(tmp_path):
    db = KuzuBackend(tmp_path / "g", embedding_dim=3)
    try:
        db.save_node(_node(embedding=[0.1, 0.2, 0.3]))
        with pytest.raises(ValueError):
            db.similarity_search([1.0, 0.0], limit=1)  # 2 != 3
    finally:
        db.close()


def test_kuzu_full_dimension_roundtrip(tmp_path):
    """A real-width vector survives storage intact — not truncated."""
    db = KuzuBackend(tmp_path / "g", embedding_dim=16)
    try:
        vec = [i / 16 for i in range(16)]
        db.save_node(_node(embedding=vec))
        loaded = db.load_node("n1")
        assert loaded["embedding"] == pytest.approx(vec)
    finally:
        db.close()


def test_kuzu_reopen_with_different_dim_raises(tmp_path):
    p = tmp_path / "g"
    db = KuzuBackend(p, embedding_dim=3)
    db.save_node(_node(embedding=[0.1, 0.2, 0.3]))
    db.close()
    with pytest.raises(ValueError):
        KuzuBackend(p, embedding_dim=8)  # persisted schema is dim-3


# --- SQLite: mismatch fails loud instead of zero-scoring ---


def test_sqlite_similarity_rejects_wrong_dim_query(tmp_path):
    db = SQLiteBackend(tmp_path / "g.db")
    try:
        db.save_node(_node(embedding=[0.1, 0.2, 0.3]))
        with pytest.raises(ValueError):
            db.similarity_search([1.0, 0.0], limit=1)
    finally:
        db.close()


# Note: the optional embedder-`dimension` hint and the MCP-server wiring
# (deriving the backend dim from the embedder, ENGRAM_* -> SYNAP_* rename,
# dropping the tests.conftest import) belong to J5 subtask 5 ("embedding model
# to an environment variable"), not this truncation fix. Deferred there.
