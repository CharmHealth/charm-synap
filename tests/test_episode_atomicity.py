"""Atomic episode writes and episode-safe eviction (J5 / CH-687).

An episode is three nodes + edges. Before this fix they were written with six
independent commits (so a mid-write failure sheared the episode) and evicted
per node (so independent decay sheared it later). The fix makes the whole
episode the atomic unit: one transaction on write, all-or-nothing on evict.

Postgres needs a live server (SYNAP_TEST_POSTGRES_DSN); it is skipped otherwise.
"""

from __future__ import annotations

import inspect
import os
import uuid

import pytest

from synap.backends.kuzu import KuzuBackend
from synap.backends.sqlite import SQLiteBackend
from synap.backends.postgres import PostgresBackend

try:
    import asyncpg
except ImportError:  # pragma: no cover
    asyncpg = None

DSN = os.environ.get("SYNAP_TEST_POSTGRES_DSN")


async def _call(backend, method: str, *args, **kwargs):
    result = getattr(backend, method)(*args, **kwargs)
    if inspect.iscoroutine(result):
        return await result
    return result


def _node(id: str, episode_id: str | None = None, utility: float = 1.0):
    meta = {"episode_id": episode_id} if episode_id else {}
    return {
        "id": id,
        "node_type": "episodic",
        "content": f"content {id}",
        "embedding": [0.1, 0.2, 0.3],
        "utility_score": utility,
        "access_count": 0,
        "created_at": "2026-01-01T00:00:00+00:00",
        "last_accessed": "2026-01-01T00:00:00+00:00",
        "metadata": meta,
    }


def _edge(id: str, source: str, target: str):
    return {
        "id": id,
        "source_id": source,
        "target_id": target,
        "relation_type": "produced",
        "weight": 1.0,
        "created_at": "2026-01-01T00:00:00+00:00",
        "metadata": {},
    }


@pytest.fixture(params=["kuzu", "sqlite", "postgres"])
async def backend(request, tmp_path):
    name = request.param
    if name == "kuzu":
        b = KuzuBackend(tmp_path / "graph", embedding_dim=3)
        try:
            yield b
        finally:
            b.close()
    elif name == "sqlite":
        b = SQLiteBackend(tmp_path / "graph.db")
        try:
            yield b
        finally:
            b.close()
    else:
        if asyncpg is None or not DSN:
            pytest.skip("set SYNAP_TEST_POSTGRES_DSN (with asyncpg) for Postgres")
        prefix = f"atom_{uuid.uuid4().hex[:12]}_"
        pool = await asyncpg.create_pool(DSN, min_size=1, max_size=5)
        b = PostgresBackend(pool, embedding_dim=3, table_prefix=prefix)
        await b.init()
        try:
            yield b
        finally:
            async with pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {prefix}edges CASCADE")
                await conn.execute(f"DROP TABLE IF EXISTS {prefix}nodes CASCADE")
            await pool.close()


async def test_write_batch_persists_nodes_and_edges(backend):
    nodes = [_node("e1_cue", "e1"), _node("e1_content", "e1"), _node("e1_outcome", "e1")]
    edges = [_edge("e1_pr", "e1_cue", "e1_content"), _edge("e1_ri", "e1_content", "e1_outcome")]

    await _call(backend, "write_batch", nodes, edges)

    assert await _call(backend, "node_count") == 3
    assert await _call(backend, "edge_count") == 2
    assert await _call(backend, "load_node", "e1_outcome") is not None


async def test_write_batch_is_atomic_on_failure(backend):
    """A batch that fails partway leaves nothing behind — no sheared episode."""
    good = _node("e1_cue", "e1")
    bad = {"id": "e1_content", "node_type": "episodic"}  # missing 'content' -> raises

    with pytest.raises(Exception):
        await _call(backend, "write_batch", [good, bad], [])

    assert await _call(backend, "node_count") == 0, "partial episode was persisted"
    assert await _call(backend, "load_node", "e1_cue") is None


async def test_delete_nodes_batch_removes_all(backend):
    for i in range(3):
        await _call(backend, "save_node", _node(f"n{i}", "e1"))
    await _call(backend, "delete_nodes_batch", ["n0", "n1", "n2"])
    assert await _call(backend, "node_count") == 0


# --- B1 eviction policy (pure) ---


def test_select_evictions_non_episodic_evicted_individually():
    from synap._utils import select_evictions

    items = [("a", 0.05, None), ("b", 0.9, None)]
    assert select_evictions(items, 0.1) == ["a"]


def test_select_evictions_keeps_episode_if_any_member_warm():
    from synap._utils import select_evictions

    items = [("cue", 0.05, "e1"), ("content", 0.05, "e1"), ("outcome", 0.9, "e1")]
    assert select_evictions(items, 0.1) == []


def test_select_evictions_removes_whole_episode_when_all_cold():
    from synap._utils import select_evictions

    items = [("cue", 0.02, "e1"), ("content", 0.03, "e1"), ("outcome", 0.01, "e1")]
    assert set(select_evictions(items, 0.1)) == {"cue", "content", "outcome"}


# --- episode-safe eviction end-to-end (Kuzu via PersistentGraph) ---


async def test_evict_keeps_episode_with_a_warm_node(tmp_path):
    from synap.persistent_graph import PersistentGraph

    backend = KuzuBackend(tmp_path / "g", embedding_dim=3)
    graph = PersistentGraph(backend=backend)
    backend.save_node(_node("e1_cue", "e1", utility=0.9))  # warm cue
    backend.save_node(_node("e1_content", "e1", utility=0.02))
    backend.save_node(_node("e1_outcome", "e1", utility=0.02))

    evicted = await graph.evict(threshold=0.1)

    assert evicted == [], "episode sheared — a warm cue should keep it whole"
    assert backend.node_count() == 3
    backend.close()


async def test_evict_removes_whole_cold_episode_but_keeps_warm_semantic(tmp_path):
    from synap.persistent_graph import PersistentGraph

    backend = KuzuBackend(tmp_path / "g2", embedding_dim=3)
    graph = PersistentGraph(backend=backend)
    for role in ("cue", "content", "outcome"):
        backend.save_node(_node(f"e1_{role}", "e1", utility=0.02))
    warm = _node("kept", episode_id=None, utility=0.9)
    warm["node_type"] = "semantic"
    backend.save_node(warm)

    evicted = await graph.evict(threshold=0.1)

    assert set(evicted) == {"e1_cue", "e1_content", "e1_outcome"}
    assert backend.load_node("kept") is not None
    backend.close()
