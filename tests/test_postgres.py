"""Tests for the PostgreSQL + pgvector backend.

Parity target: these mirror ``test_kuzu.py`` over the storage contract both
backends share, plus the Postgres-only behaviours the backend adds (real
pgvector search with null/dimension-mismatch handling, FK cascade, pooled
concurrency).

A live Postgres with the ``vector`` extension is required. Point
``SYNAP_TEST_POSTGRES_DSN`` at one to run the suite; it is skipped otherwise.
CI provides a pgvector service container and sets the DSN, so the suite never
silently skips there.
"""

from __future__ import annotations

import asyncio
import os

import pytest

asyncpg = pytest.importorskip("asyncpg")

from synap.backends.postgres import PostgresBackend

DSN = os.environ.get("SYNAP_TEST_POSTGRES_DSN")

pytestmark = pytest.mark.skipif(
    not DSN, reason="set SYNAP_TEST_POSTGRES_DSN to run the Postgres backend suite"
)


def _make_node(
    id: str = "n1",
    node_type: str = "semantic",
    content: str = "test fact",
    embedding: list[float] | None = None,
):
    return {
        "id": id,
        "node_type": node_type,
        "content": content,
        "embedding": embedding or [0.1, 0.2, 0.3],
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


def _prefix(request) -> str:
    """A unique table prefix per test so tests are isolated without truncation."""
    raw = request.node.name
    safe = "".join(c if c.isalnum() else "_" for c in raw).lower()
    return f"t_{safe}_"[:48]


@pytest.fixture
async def db(request):
    prefix = _prefix(request)
    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=5)
    backend = PostgresBackend(pool, embedding_dim=3, table_prefix=prefix)
    await backend.init()
    try:
        yield backend
    finally:
        async with pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {prefix}edges CASCADE")
            await conn.execute(f"DROP TABLE IF EXISTS {prefix}nodes CASCADE")
        await pool.close()


# --- CRUD (subtask 2) ---


async def test_save_and_load_node(db: PostgresBackend):
    await db.save_node(_make_node())
    loaded = await db.load_node("n1")
    assert loaded is not None
    assert loaded["content"] == "test fact"
    assert loaded["node_type"] == "semantic"


async def test_load_nonexistent_node(db: PostgresBackend):
    assert await db.load_node("nonexistent") is None


async def test_save_and_load_edge(db: PostgresBackend):
    await db.save_node(_make_node("n1"))
    await db.save_node(_make_node("n2", content="another fact"))
    await db.save_edge(_make_edge())

    edges = await db.load_edges("n1")
    assert len(edges) == 1
    assert edges[0]["relation_type"] == "related_to"


async def test_load_edges_filtered_by_type(db: PostgresBackend):
    await db.save_node(_make_node("n1"))
    await db.save_node(_make_node("n2"))
    await db.save_edge({
        "id": "e1", "source_id": "n1", "target_id": "n2",
        "relation_type": "causes", "weight": 1.0,
        "created_at": "2026-01-01T00:00:00Z", "metadata": {},
    })
    await db.save_edge({
        "id": "e2", "source_id": "n1", "target_id": "n2",
        "relation_type": "related_to", "weight": 1.0,
        "created_at": "2026-01-01T00:00:00Z", "metadata": {},
    })

    causal = await db.load_edges("n1", edge_type="causes")
    assert len(causal) == 1
    assert causal[0]["relation_type"] == "causes"


async def test_upsert_node(db: PostgresBackend):
    await db.save_node(_make_node("n1", content="original"))
    await db.save_node(_make_node("n1", content="updated"))

    loaded = await db.load_node("n1")
    assert loaded["content"] == "updated"

    all_nodes = await db.query_nodes()
    assert len(all_nodes) == 1


# --- Query, filter, counts (subtask 3) ---


async def test_query_nodes_by_type(db: PostgresBackend):
    await db.save_node(_make_node("n1", node_type="semantic"))
    await db.save_node(_make_node("n2", node_type="episodic"))
    await db.save_node(_make_node("n3", node_type="semantic"))

    semantic = await db.query_nodes(node_type="semantic")
    assert len(semantic) == 2

    episodic = await db.query_nodes(node_type="episodic")
    assert len(episodic) == 1


async def test_query_nodes_with_filters(db: PostgresBackend):
    node1 = _make_node("n1")
    node1["metadata"] = {"tag": "important"}
    await db.save_node(node1)

    node2 = _make_node("n2")
    node2["metadata"] = {"tag": "trivial"}
    await db.save_node(node2)

    results = await db.query_nodes(filters={"tag": "important"})
    assert len(results) == 1
    assert results[0]["id"] == "n1"


async def test_node_count(db: PostgresBackend):
    await db.save_node(_make_node("n1", node_type="semantic"))
    await db.save_node(_make_node("n2", node_type="episodic"))

    assert await db.node_count() == 2
    assert await db.node_count(node_type="semantic") == 1


async def test_edge_count(db: PostgresBackend):
    await db.save_node(_make_node("n1"))
    await db.save_node(_make_node("n2"))
    await db.save_edge(_make_edge("e1", "n1", "n2"))
    await db.save_edge({
        "id": "e2", "source_id": "n2", "target_id": "n1",
        "relation_type": "causes", "weight": 1.0,
        "created_at": "2026-01-01T00:00:00Z", "metadata": {},
    })

    assert await db.edge_count() == 2
    assert await db.edge_count(relation_type="causes") == 1


# --- Vector similarity (subtask 4) ---


async def test_similarity_search(db: PostgresBackend):
    await db.save_node(_make_node("n1", embedding=[1.0, 0.0, 0.0]))
    await db.save_node(_make_node("n2", embedding=[0.0, 1.0, 0.0]))
    await db.save_node(_make_node("n3", embedding=[0.9, 0.1, 0.0]))

    results = await db.similarity_search([1.0, 0.0, 0.0], limit=2)
    assert len(results) == 2
    assert results[0]["id"] == "n1"
    assert results[1]["id"] == "n3"


async def test_similarity_search_by_type(db: PostgresBackend):
    await db.save_node(
        _make_node("n1", node_type="semantic", embedding=[1.0, 0.0, 0.0])
    )
    await db.save_node(
        _make_node("n2", node_type="episodic", embedding=[1.0, 0.0, 0.0])
    )

    results = await db.similarity_search([1.0, 0.0, 0.0], node_type="semantic")
    assert len(results) == 1
    assert results[0]["node_type"] == "semantic"


async def test_similarity_search_excludes_null_embedding(db: PostgresBackend):
    """Nodes stored without an embedding must not surface in vector search."""
    await db.save_node(_make_node("with_vec", embedding=[1.0, 0.0, 0.0]))

    no_vec = _make_node("no_vec")
    no_vec["embedding"] = None
    await db.save_node(no_vec)

    results = await db.similarity_search([1.0, 0.0, 0.0], limit=10)
    ids = {r["id"] for r in results}
    assert "with_vec" in ids
    assert "no_vec" not in ids


async def test_similarity_search_dimension_mismatch(db: PostgresBackend):
    """A query vector of the wrong dimension is rejected by pgvector."""
    await db.save_node(_make_node("n1", embedding=[1.0, 0.0, 0.0]))

    with pytest.raises(Exception):
        await db.similarity_search([1.0, 0.0], limit=1)  # dim 2 vs configured 3


# --- Graph traversal (subtask 5) ---


async def test_traverse_basic(db: PostgresBackend):
    await db.save_node(_make_node("n1"))
    await db.save_node(_make_node("n2"))
    await db.save_node(_make_node("n3"))
    await db.save_edge(_make_edge("e1", "n1", "n2"))
    await db.save_edge(_make_edge("e2", "n2", "n3"))

    result = await db.traverse("n1", max_depth=1)
    ids = {r["id"] for r in result}
    assert "n2" in ids
    assert "n3" not in ids

    result = await db.traverse("n1", max_depth=2)
    ids = {r["id"] for r in result}
    assert "n2" in ids
    assert "n3" in ids


async def test_traverse_max_nodes(db: PostgresBackend):
    await db.save_node(_make_node("center"))
    for i in range(10):
        await db.save_node(_make_node(f"spoke_{i}"))
        await db.save_edge(_make_edge(f"e_{i}", "center", f"spoke_{i}"))

    result = await db.traverse("center", max_nodes=3)
    assert len(result) == 3


async def test_traverse_bidirectional(db: PostgresBackend):
    await db.save_node(_make_node("n1"))
    await db.save_node(_make_node("n2"))
    await db.save_edge(_make_edge("e1", "n1", "n2"))

    result = await db.traverse("n2", max_depth=1)
    ids = {r["id"] for r in result}
    assert "n1" in ids


async def test_traverse_excludes_start(db: PostgresBackend):
    await db.save_node(_make_node("n1"))
    await db.save_node(_make_node("n2"))
    await db.save_edge(_make_edge("e1", "n1", "n2"))

    result = await db.traverse("n1", max_depth=2)
    ids = {r["id"] for r in result}
    assert "n1" not in ids


async def test_traverse_edge_type_filter(db: PostgresBackend):
    await db.save_node(_make_node("n1"))
    await db.save_node(_make_node("n2"))
    await db.save_node(_make_node("n3"))
    await db.save_edge({
        "id": "e1", "source_id": "n1", "target_id": "n2",
        "relation_type": "causes", "weight": 1.0,
        "created_at": "2026-01-01T00:00:00Z", "metadata": {},
    })
    await db.save_edge({
        "id": "e2", "source_id": "n1", "target_id": "n3",
        "relation_type": "related_to", "weight": 1.0,
        "created_at": "2026-01-01T00:00:00Z", "metadata": {},
    })

    result = await db.traverse("n1", edge_types=["causes"], max_depth=1)
    ids = {r["id"] for r in result}
    assert ids == {"n2"}


async def test_traverse_cycle_safety(db: PostgresBackend):
    """A cycle must not send the recursive CTE into an infinite loop."""
    await db.save_node(_make_node("a"))
    await db.save_node(_make_node("b"))
    await db.save_node(_make_node("c"))
    await db.save_edge(_make_edge("e1", "a", "b"))
    await db.save_edge(_make_edge("e2", "b", "c"))
    await db.save_edge(_make_edge("e3", "c", "a"))  # closes the loop

    result = await db.traverse("a", max_depth=5)
    ids = {r["id"] for r in result}
    assert ids == {"b", "c"}  # terminates, excludes start, deduped


# --- Delete / cascade, timestamp, metadata, concurrency (subtask 6) ---


async def test_delete_node(db: PostgresBackend):
    await db.save_node(_make_node("n1"))
    await db.save_node(_make_node("n2"))
    await db.save_edge(_make_edge("e1", "n1", "n2"))

    await db.delete_node("n1")
    assert await db.load_node("n1") is None


async def test_delete_edge(db: PostgresBackend):
    await db.save_node(_make_node("n1"))
    await db.save_node(_make_node("n2"))
    await db.save_edge(_make_edge("e1", "n1", "n2"))

    await db.delete_edge("e1")
    assert len(await db.load_edges("n1")) == 0


async def test_delete_node_cascades_edges(db: PostgresBackend):
    """The FK ``ON DELETE CASCADE`` removes a node's edges with it."""
    await db.save_node(_make_node("n1"))
    await db.save_node(_make_node("n2"))
    await db.save_edge(_make_edge("e1", "n1", "n2"))

    await db.delete_node("n1")
    assert len(await db.load_edges("n2")) == 0
    assert await db.edge_count() == 0


async def test_timestamp_roundtrip(db: PostgresBackend):
    """ISO-string timestamps from higher layers survive the timestamptz column."""
    node = _make_node("n1")
    node["created_at"] = "2026-03-14T09:26:53+00:00"
    node["last_accessed"] = "2026-03-14T09:26:53+00:00"
    await db.save_node(node)

    loaded = await db.load_node("n1")
    from datetime import datetime

    assert datetime.fromisoformat(loaded["created_at"]) == datetime.fromisoformat(
        "2026-03-14T09:26:53+00:00"
    )


async def test_metadata_roundtrip(db: PostgresBackend):
    """Nested metadata round-trips through JSONB intact."""
    node = _make_node("n1")
    node["metadata"] = {"tag": "x", "nested": {"a": 1, "b": [1, 2, 3]}, "flag": True}
    await db.save_node(node)

    loaded = await db.load_node("n1")
    assert loaded["metadata"] == {
        "tag": "x",
        "nested": {"a": 1, "b": [1, 2, 3]},
        "flag": True,
    }


async def test_concurrent_writes(db: PostgresBackend):
    """Independent writes across pooled connections all land, no corruption."""
    await asyncio.gather(
        *(db.save_node(_make_node(f"n{i}")) for i in range(20))
    )
    assert await db.node_count() == 20


async def test_concurrent_upserts_converge(db: PostgresBackend):
    """Concurrent upserts to the same id converge to a single row."""
    await asyncio.gather(
        *(db.save_node(_make_node("same", content=f"v{i}")) for i in range(20))
    )
    assert await db.node_count() == 1
    loaded = await db.load_node("same")
    assert loaded["content"].startswith("v")


async def test_persistence_across_backend_instances(db: PostgresBackend):
    """A second backend over the same pool/tables sees committed data."""
    await db.save_node(_make_node("n1"))

    other = PostgresBackend(
        db._pool, embedding_dim=3, table_prefix=db._prefix
    )
    loaded = await other.load_node("n1")
    assert loaded is not None
    assert loaded["content"] == "test fact"


# --- Vector index (J8) ---


async def test_init_creates_an_hnsw_index_on_the_embedding(db: PostgresBackend):
    """Without an index for `<=>`, every similarity search scans every node."""
    # Looked up by definition rather than by name: Postgres truncates
    # identifiers at 63 bytes, and this suite's per-test table prefix is long
    # enough to clip the tail off the intended name.
    async with db._pool.acquire() as conn:
        definition = await conn.fetchval(
            "SELECT indexdef FROM pg_indexes "
            "WHERE tablename = $1 AND indexdef LIKE '%(embedding%'",
            f"{db._prefix}nodes",
        )
    assert definition is not None, "no index was created on the embedding column"
    assert "USING hnsw" in definition, definition
    # vector_cosine_ops, because search_similar orders by <=>. An l2 index looks
    # perfectly healthy in pg_indexes and simply never gets used.
    assert "vector_cosine_ops" in definition, definition


async def test_the_planner_uses_that_index_for_the_search_query(db: PostgresBackend):
    """Proves the operator class matches the operator the backend actually uses.

    This is the assertion that catches a wrong opclass. An index built with
    vector_l2_ops exists, reports healthy, and cannot answer an ORDER BY on
    `<=>` — the planner just falls back to a sequential scan and nothing errors.
    Turning seqscan off makes the difference observable: a usable index shows up
    in the plan, a mismatched one leaves a Seq Scan behind.
    """
    await db.save_node(_make_node("n1", embedding=[1.0, 0.0, 0.0]))
    async with db._pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute("SET LOCAL enable_seqscan = off")
            rows = await conn.fetch(
                f"EXPLAIN SELECT id FROM {db._prefix}nodes "
                f"ORDER BY embedding <=> '[1,0,0]'::vector LIMIT 5"
            )
    plan = "\n".join(r["QUERY PLAN"] for r in rows)
    # Not asserted by index name — see the note in the test above.
    assert "Index Scan" in plan, plan
    assert "embedding <=>" in plan, plan
    assert "Seq Scan" not in plan, plan
