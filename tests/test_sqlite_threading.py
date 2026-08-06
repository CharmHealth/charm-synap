"""SQLite backend must survive PersistentGraph's cross-thread dispatch (CH-728).

PersistentGraph runs sync backends via ``asyncio.to_thread``, so backend calls
land on arbitrary worker threads of the default executor — not the thread that
constructed the backend. ``sqlite3.connect`` defaults to ``check_same_thread=
True``, so the shared connection raises ``ProgrammingError`` the first time a
call runs on a different thread. That makes SQLite + PersistentGraph crash on
first write.
"""

from __future__ import annotations

import asyncio

from synap.backends.sqlite import SQLiteBackend
from synap.persistent_graph import PersistentGraph
from synap.types import MemoryNode, MemoryType


def _node(node_id: str) -> MemoryNode:
    return MemoryNode(
        id=node_id,
        node_type=MemoryType.SEMANTIC,
        content=f"content {node_id}",
        embedding=[0.1, 0.2, 0.3],
    )


async def test_save_then_load_across_threads(tmp_path):
    backend = SQLiteBackend(tmp_path / "g.db")
    graph = PersistentGraph(backend=backend)
    try:
        await graph.add_node(_node("n1"))  # dispatched onto an executor thread
        loaded = await graph.get_node("n1")
        assert loaded is not None
        assert loaded.content == "content n1"
    finally:
        backend.close()


async def test_concurrent_writes_across_threads(tmp_path):
    """Several writes dispatched together land on different worker threads."""
    backend = SQLiteBackend(tmp_path / "g.db")
    graph = PersistentGraph(backend=backend)
    try:
        await asyncio.gather(*(graph.add_node(_node(f"n{i}")) for i in range(8)))
        assert await graph.node_count() == 8
    finally:
        backend.close()


def test_traverse_does_not_deadlock_on_reentrant_lock(tmp_path):
    """traverse holds the lock and calls load_edges/load_node under it.

    The serialization lock must be reentrant; a plain Lock would deadlock here
    the instant traverse re-enters. This pins that choice.
    """
    backend = SQLiteBackend(tmp_path / "g.db")
    try:
        for nid in ("a", "b", "c"):
            backend.save_node({**_node_dict(nid)})
        backend.save_edge(_edge_dict("a_b", "a", "b"))
        backend.save_edge(_edge_dict("b_c", "b", "c"))

        reached = backend.traverse("a", max_depth=2, max_nodes=10)

        assert {n["id"] for n in reached} == {"b", "c"}
    finally:
        backend.close()


def _node_dict(node_id: str) -> dict:
    return {
        "id": node_id,
        "node_type": "semantic",
        "content": f"content {node_id}",
        "embedding": [0.1, 0.2, 0.3],
        "utility_score": 1.0,
        "access_count": 0,
        "created_at": "2026-01-01T00:00:00+00:00",
        "last_accessed": "2026-01-01T00:00:00+00:00",
        "metadata": {},
    }


def _edge_dict(edge_id: str, source: str, target: str) -> dict:
    return {
        "id": edge_id,
        "source_id": source,
        "target_id": target,
        "relation_type": "relates_to",
        "weight": 1.0,
        "created_at": "2026-01-01T00:00:00+00:00",
        "metadata": {},
    }
