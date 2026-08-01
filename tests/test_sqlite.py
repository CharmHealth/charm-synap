"""Tests for SQLite storage backend."""

import tempfile
from pathlib import Path

import pytest

from synap.backends.sqlite import SQLiteBackend


@pytest.fixture
def db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    backend = SQLiteBackend(path)
    yield backend
    backend.close()
    Path(path).unlink(missing_ok=True)


def _make_node(id: str = "n1", node_type: str = "semantic", content: str = "test fact"):
    return {
        "id": id,
        "node_type": node_type,
        "content": content,
        "embedding": [0.1, 0.2, 0.3],
        "utility_score": 1.0,
        "access_count": 0,
        "created_at": "2026-01-01T00:00:00Z",
        "last_accessed": "2026-01-01T00:00:00Z",
        "metadata": {"tag": "test"},
    }


def _make_edge(id: str = "e1", source: str = "n1", target: str = "n2"):
    return {
        "id": id,
        "source_id": source,
        "target_id": target,
        "relation_type": "related_to",
        "weight": 1.0,
        "created_at": "2026-01-01T00:00:00Z",
        "metadata": {},
    }


def test_save_and_load_node(db: SQLiteBackend):
    node = _make_node()
    db.save_node(node)

    loaded = db.load_node("n1")
    assert loaded is not None
    assert loaded["content"] == "test fact"
    assert loaded["node_type"] == "semantic"


def test_load_nonexistent_node(db: SQLiteBackend):
    assert db.load_node("nonexistent") is None


def test_save_and_load_edge(db: SQLiteBackend):
    db.save_node(_make_node("n1"))
    db.save_node(_make_node("n2", content="another fact"))
    db.save_edge(_make_edge())

    edges = db.load_edges("n1")
    assert len(edges) == 1
    assert edges[0]["relation_type"] == "related_to"


def test_load_edges_filtered_by_type(db: SQLiteBackend):
    db.save_node(_make_node("n1"))
    db.save_node(_make_node("n2"))
    db.save_edge({"id": "e1", "source_id": "n1", "target_id": "n2",
                  "relation_type": "causes", "weight": 1.0,
                  "created_at": "2026-01-01T00:00:00Z", "metadata": {}})
    db.save_edge({"id": "e2", "source_id": "n1", "target_id": "n2",
                  "relation_type": "related_to", "weight": 1.0,
                  "created_at": "2026-01-01T00:00:00Z", "metadata": {}})

    causal = db.load_edges("n1", edge_type="causes")
    assert len(causal) == 1
    assert causal[0]["relation_type"] == "causes"


def test_query_nodes_by_type(db: SQLiteBackend):
    db.save_node(_make_node("n1", node_type="semantic"))
    db.save_node(_make_node("n2", node_type="episodic"))
    db.save_node(_make_node("n3", node_type="semantic"))

    semantic = db.query_nodes(node_type="semantic")
    assert len(semantic) == 2

    episodic = db.query_nodes(node_type="episodic")
    assert len(episodic) == 1


def test_query_nodes_with_filters(db: SQLiteBackend):
    node = _make_node("n1")
    node["metadata"] = {"tag": "important"}
    db.save_node(node)

    node2 = _make_node("n2")
    node2["metadata"] = {"tag": "trivial"}
    db.save_node(node2)

    results = db.query_nodes(filters={"tag": "important"})
    assert len(results) == 1
    assert results[0]["id"] == "n1"


def test_delete_node(db: SQLiteBackend):
    db.save_node(_make_node("n1"))
    db.save_node(_make_node("n2"))
    db.save_edge(_make_edge("e1", "n1", "n2"))

    db.delete_node("n1")
    assert db.load_node("n1") is None
    # Edge should also be deleted
    assert len(db.load_edges("n2")) == 0


def test_delete_edge(db: SQLiteBackend):
    db.save_node(_make_node("n1"))
    db.save_node(_make_node("n2"))
    db.save_edge(_make_edge("e1", "n1", "n2"))

    db.delete_edge("e1")
    assert len(db.load_edges("n1")) == 0


def test_similarity_search(db: SQLiteBackend):
    node1 = _make_node("n1")
    node1["embedding"] = [1.0, 0.0, 0.0]
    db.save_node(node1)

    node2 = _make_node("n2")
    node2["embedding"] = [0.0, 1.0, 0.0]
    db.save_node(node2)

    node3 = _make_node("n3")
    node3["embedding"] = [0.9, 0.1, 0.0]
    db.save_node(node3)

    # Search for vector close to [1, 0, 0]
    results = db.similarity_search([1.0, 0.0, 0.0], limit=2)
    assert len(results) == 2
    # n1 should be most similar, then n3
    assert results[0]["id"] == "n1"
    assert results[1]["id"] == "n3"


def test_similarity_search_by_type(db: SQLiteBackend):
    node1 = _make_node("n1", node_type="semantic")
    node1["embedding"] = [1.0, 0.0, 0.0]
    db.save_node(node1)

    node2 = _make_node("n2", node_type="episodic")
    node2["embedding"] = [1.0, 0.0, 0.0]
    db.save_node(node2)

    results = db.similarity_search([1.0, 0.0, 0.0], node_type="semantic")
    assert len(results) == 1
    assert results[0]["node_type"] == "semantic"


def test_upsert_node(db: SQLiteBackend):
    """save_node with same ID should update, not duplicate."""
    db.save_node(_make_node("n1", content="original"))
    db.save_node(_make_node("n1", content="updated"))

    loaded = db.load_node("n1")
    assert loaded["content"] == "updated"

    all_nodes = db.query_nodes()
    assert len(all_nodes) == 1


def test_persistence(tmp_path: Path):
    """Data survives close and reopen."""
    db_path = tmp_path / "test.db"

    backend1 = SQLiteBackend(db_path)
    backend1.save_node(_make_node("n1"))
    backend1.close()

    backend2 = SQLiteBackend(db_path)
    loaded = backend2.load_node("n1")
    assert loaded is not None
    assert loaded["content"] == "test fact"
    backend2.close()
