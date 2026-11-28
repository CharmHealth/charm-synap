"""Opens a psycopg pool for the Postgres tests (needs the postgres extra)."""

from __future__ import annotations

try:
    from psycopg_pool import AsyncConnectionPool
except ImportError:  # pragma: no cover - exercised via skip
    AsyncConnectionPool = None  # type: ignore[assignment,misc]

AVAILABLE = AsyncConnectionPool is not None


async def open_pool(dsn: str) -> AsyncConnectionPool:
    """A small pool, opened, as an application would hand to PostgresBackend.

    No autocommit or row-factory settings on purpose: the backend must work on a
    pool with psycopg's defaults, not only on one configured the way it likes.
    """
    pool = AsyncConnectionPool(dsn, min_size=1, max_size=5, open=False)
    await pool.open(wait=True, timeout=30)
    return pool
