"""Cross-backend: valid_from/valid_until survive persistence (J5 / CH-687).

Fact expiry is expressed by a node's validity window. The domain layer
serializes `valid_from`/`valid_until`, but before this fix the Kuzu and Postgres
schemas had no columns for them, so they were silently dropped on save and a
fact's expiry did not survive a restart. SQLite stores the whole node as a JSON
blob and already preserved them. This pins the behavior on all three.

Postgres needs a live server (SYNAP_TEST_POSTGRES_DSN); it is skipped otherwise.
"""

from __future__ import annotations

import inspect
import os
import uuid
from datetime import datetime

import pytest

from synap.backends.kuzu import KuzuBackend
from synap.backends.postgres import PostgresBackend
from synap.backends.sqlite import SQLiteBackend

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


def _node(id: str = "n1", valid_from=None, valid_until=None):
    return {
        "id": id,
        "node_type": "semantic",
        "content": "a fact with an expiry",
        "embedding": [0.1, 0.2, 0.3],
        "utility_score": 1.0,
        "access_count": 0,
        "created_at": "2026-01-01T00:00:00+00:00",
        "last_accessed": "2026-01-01T00:00:00+00:00",
        "metadata": {},
        "valid_from": valid_from,
        "valid_until": valid_until,
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
        prefix = f"tmp_{uuid.uuid4().hex[:12]}_"
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


async def test_validity_window_survives_roundtrip(backend):
    await _call(
        backend,
        "save_node",
        _node(
            valid_from="2026-01-01T00:00:00+00:00",
            valid_until="2026-12-31T00:00:00+00:00",
        ),
    )
    loaded = await _call(backend, "load_node", "n1")

    assert loaded["valid_from"] is not None, "valid_from was dropped on persist"
    assert loaded["valid_until"] is not None, "valid_until was dropped on persist"
    assert datetime.fromisoformat(loaded["valid_from"]) == datetime.fromisoformat(
        "2026-01-01T00:00:00+00:00"
    )
    assert datetime.fromisoformat(loaded["valid_until"]) == datetime.fromisoformat(
        "2026-12-31T00:00:00+00:00"
    )


async def test_null_validity_window_roundtrips(backend):
    """A node with no expiry round-trips as None, not a fabricated timestamp."""
    await _call(backend, "save_node", _node())
    loaded = await _call(backend, "load_node", "n1")
    assert loaded.get("valid_from") is None
    assert loaded.get("valid_until") is None


def test_kuzu_validity_window_survives_restart(tmp_path):
    """The deliverable's exact claim: fact expiry survives a restart.

    Reopen the on-disk store with a fresh backend instance and confirm the
    validity window is still there (not just in-instance state).
    """
    path = tmp_path / "restart"
    b1 = KuzuBackend(path, embedding_dim=3)
    b1.save_node(
        _node(
            valid_from="2026-01-01T00:00:00+00:00",
            valid_until="2026-06-01T00:00:00+00:00",
        )
    )
    b1.close()

    b2 = KuzuBackend(path, embedding_dim=3)
    loaded = b2.load_node("n1")
    b2.close()

    assert loaded is not None
    assert datetime.fromisoformat(loaded["valid_until"]) == datetime.fromisoformat(
        "2026-06-01T00:00:00+00:00"
    )
