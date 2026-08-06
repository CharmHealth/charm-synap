"""Cross-backend behavioural parity.

One query, one result set — regardless of which backend answers it. Each
backend implements ``query_nodes`` differently (Cypher, SQL, recursive push),
but the observable contract must not differ. These tests run the same body
against every backend.

The load-bearing case is filters-before-LIMIT: metadata filters must be
applied *before* the row limit, so a query cannot drop matching rows just
because higher-utility non-matching rows filled the limit window first.
Postgres pushes the filter into SQL and already satisfies this; Kuzu and
SQLite filtered in Python *after* the limit and are fixed to match.
"""

from __future__ import annotations

import inspect
import os
import uuid

import pytest

from synap.backends.kuzu import KuzuBackend
from synap.backends.postgres import PostgresBackend
from synap.backends.sqlite import SQLiteBackend

try:
    import asyncpg
except ImportError:  # pragma: no cover - exercised via skip
    asyncpg = None

DSN = os.environ.get("SYNAP_TEST_POSTGRES_DSN")


async def _call(backend, method: str, *args, **kwargs):
    """Invoke a backend method uniformly, awaiting async backends."""
    result = getattr(backend, method)(*args, **kwargs)
    if inspect.iscoroutine(result):
        return await result
    return result


def _node(id: str, tag: str, utility: float):
    return {
        "id": id,
        "node_type": "semantic",
        "content": f"content {id}",
        "embedding": [0.1, 0.2, 0.3],
        "utility_score": utility,
        "access_count": 0,
        "created_at": "2026-01-01T00:00:00Z",
        "last_accessed": "2026-01-01T00:00:00Z",
        "metadata": {"tag": tag},
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
            pytest.skip("set SYNAP_TEST_POSTGRES_DSN (with asyncpg) for Postgres parity")
        prefix = f"c_{uuid.uuid4().hex[:12]}_"
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


async def test_filters_apply_before_limit(backend):
    """Filters must be applied before the limit, on every backend.

    Six matching nodes sit at *low* utility; six non-matching nodes sit at
    *high* utility. With limit=5 and ORDER BY utility DESC, a backend that
    limits before filtering grabs the six high-utility non-matches, then
    filters them all away and returns nothing. Filtering first returns five
    matches.
    """
    for i in range(6):
        await _call(backend, "save_node", _node(f"keep{i}", "keep", utility=0.1 + i * 0.01))
    for i in range(6):
        await _call(backend, "save_node", _node(f"drop{i}", "drop", utility=0.9 + i * 0.01))

    results = await _call(backend, "query_nodes", filters={"tag": "keep"}, limit=5)

    assert len(results) == 5
    assert all(r["metadata"]["tag"] == "keep" for r in results)


async def test_query_limit_bounds_result_set(backend):
    """The limit still caps the result set once filters have been applied."""
    for i in range(10):
        await _call(backend, "save_node", _node(f"keep{i}", "keep", utility=0.5))

    results = await _call(backend, "query_nodes", filters={"tag": "keep"}, limit=3)
    assert len(results) == 3
    assert all(r["metadata"]["tag"] == "keep" for r in results)
