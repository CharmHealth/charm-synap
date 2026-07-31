"""Active procedures are immune to eviction; retired ones evict (CH-728 item 4).

A procedure is a learned capability, expensive to relearn, and its active set is
naturally bounded (one per task_type). So an active procedure is never evicted,
even below the utility threshold. A retired (superseded) procedure is not
special — it fades like any other node. Other node types evict normally.
"""

from __future__ import annotations

from synap.backends.sqlite import SQLiteBackend
from synap.graph import MemoryGraph
from synap.persistent_graph import PersistentGraph
from synap.types import MemoryNode, MemoryType


def _node(nid, node_type, utility, superseded=False):
    meta = {"superseded": True} if superseded else {}
    return MemoryNode(
        id=nid,
        node_type=node_type,
        content=f"c-{nid}",
        embedding=[0.1, 0.2, 0.3],
        utility_score=utility,
        metadata=meta,
    )


async def test_active_procedure_below_threshold_is_not_evicted():
    g = MemoryGraph()
    await g.add_node(_node("p", MemoryType.PROCEDURAL, 0.02))
    evicted = await g.evict(threshold=0.1)
    assert "p" not in evicted
    assert await g.get_node("p") is not None


async def test_retired_procedure_below_threshold_is_evicted():
    g = MemoryGraph()
    await g.add_node(_node("p", MemoryType.PROCEDURAL, 0.02, superseded=True))
    evicted = await g.evict(threshold=0.1)
    assert "p" in evicted


async def test_semantic_below_threshold_still_evicts():
    g = MemoryGraph()
    await g.add_node(_node("s", MemoryType.SEMANTIC, 0.02))
    evicted = await g.evict(threshold=0.1)
    assert "s" in evicted


def _dict(nid, node_type, utility, superseded=False):
    meta = {"superseded": True} if superseded else {}
    return {
        "id": nid,
        "node_type": node_type,
        "content": f"c-{nid}",
        "embedding": [0.1, 0.2, 0.3],
        "utility_score": utility,
        "access_count": 0,
        "created_at": "2026-01-01T00:00:00+00:00",
        "last_accessed": "2026-01-01T00:00:00+00:00",
        "metadata": meta,
    }


async def test_active_procedure_protected_over_persistent_graph(tmp_path):
    backend = SQLiteBackend(tmp_path / "g.db")
    graph = PersistentGraph(backend=backend)
    try:
        backend.save_node(_dict("active", "procedural", 0.02))
        backend.save_node(_dict("retired", "procedural", 0.02, superseded=True))
        backend.save_node(_dict("fact", "semantic", 0.02))

        evicted = await graph.evict(threshold=0.1)

        assert "active" not in evicted
        assert backend.load_node("active") is not None
        assert set(evicted) == {"retired", "fact"}
    finally:
        backend.close()
