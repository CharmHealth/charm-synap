"""Idempotent forward-migration of stores created before a column existed (J5).

We assume a fresh schema for new stores, but a store created before the
temporal columns were added must not hard-fail on every operation — opening it
with the current code adds the missing columns via idempotent ALTER (additive
only; a versioned migration path is ROADMAP task 7).

Postgres needs a live server (SYNAP_TEST_POSTGRES_DSN); it is skipped otherwise.
"""

from __future__ import annotations

import gc
import os
import uuid

import pytest

from synap.backends.kuzu import KuzuBackend
from synap.backends.postgres import PostgresBackend

try:
    import asyncpg
except ImportError:  # pragma: no cover
    asyncpg = None

DSN = os.environ.get("SYNAP_TEST_POSTGRES_DSN")


def _node_with_validity(node_id: str = "n1"):
    return {
        "id": node_id,
        "node_type": "semantic",
        "content": "c",
        "embedding": [0.1, 0.2, 0.3],
        "utility_score": 1.0,
        "access_count": 0,
        "created_at": "2026-01-01T00:00:00+00:00",
        "last_accessed": "2026-01-01T00:00:00+00:00",
        "metadata": {},
        "valid_from": "2026-01-01T00:00:00+00:00",
        "valid_until": "2026-12-31T00:00:00+00:00",
    }


def test_kuzu_migrates_store_missing_temporal_columns(tmp_path):
    import kuzu

    path = str(tmp_path / "old_store")
    db = kuzu.Database(path)
    conn = kuzu.Connection(db)
    # Pre-branch schema: no valid_from / valid_until.
    conn.execute(
        "CREATE NODE TABLE MemoryNode("
        "id STRING, node_type STRING, content STRING, embedding DOUBLE[3], "
        "utility_score DOUBLE, access_count INT64, created_at STRING, "
        "last_accessed STRING, metadata STRING, PRIMARY KEY(id))"
    )
    conn.execute(
        "CREATE REL TABLE MemoryEdge(FROM MemoryNode TO MemoryNode, id STRING, "
        "relation_type STRING, weight DOUBLE, created_at STRING, metadata STRING)"
    )
    del conn
    del db
    gc.collect()  # release the single-writer lock before reopening

    # Reopening with the current backend must add the missing columns.
    backend = KuzuBackend(path, embedding_dim=3)
    backend.save_node(_node_with_validity())
    loaded = backend.load_node("n1")
    backend.close()

    assert loaded is not None
    assert loaded["valid_from"] is not None
    assert loaded["valid_until"] is not None


@pytest.mark.skipif(
    asyncpg is None or not DSN,
    reason="set SYNAP_TEST_POSTGRES_DSN (with asyncpg) for Postgres migration test",
)
async def test_postgres_migrates_store_missing_temporal_columns():
    prefix = f"mig_{uuid.uuid4().hex[:12]}_"
    pool = await asyncpg.create_pool(DSN, min_size=1, max_size=5)
    async with pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # Pre-branch schema: no valid_from / valid_until.
        await conn.execute(
            f"CREATE TABLE {prefix}nodes ("
            f"id TEXT PRIMARY KEY, node_type TEXT NOT NULL, content TEXT NOT NULL, "
            f"embedding vector(3), utility_score DOUBLE PRECISION, "
            f"access_count INTEGER, created_at TIMESTAMPTZ NOT NULL, "
            f"last_accessed TIMESTAMPTZ NOT NULL, metadata JSONB)"
        )
        await conn.execute(
            f"CREATE TABLE {prefix}edges ("
            f"id TEXT PRIMARY KEY, "
            f"source_id TEXT NOT NULL REFERENCES {prefix}nodes(id) ON DELETE CASCADE, "
            f"target_id TEXT NOT NULL REFERENCES {prefix}nodes(id) ON DELETE CASCADE, "
            f"relation_type TEXT NOT NULL, weight DOUBLE PRECISION, "
            f"created_at TIMESTAMPTZ NOT NULL, metadata JSONB)"
        )

    backend = PostgresBackend(pool, embedding_dim=3, table_prefix=prefix)
    await backend.init()  # runs the idempotent ALTER ... ADD COLUMN IF NOT EXISTS
    await backend.save_node(_node_with_validity())
    loaded = await backend.load_node("n1")

    async with pool.acquire() as conn:
        await conn.execute(f"DROP TABLE IF EXISTS {prefix}edges CASCADE")
        await conn.execute(f"DROP TABLE IF EXISTS {prefix}nodes CASCADE")
    await pool.close()

    assert loaded is not None
    assert loaded["valid_from"] is not None
    assert loaded["valid_until"] is not None
