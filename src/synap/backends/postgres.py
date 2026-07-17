"""PostgreSQL storage backend — async, multi-process safe, pgvector similarity."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import asyncpg


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_timestamp(value: Any) -> datetime:
    """Coerce a caller-supplied timestamp into a datetime asyncpg can encode.

    Synap's higher layers serialize timestamps as ISO strings so they round-trip
    through any backend, but asyncpg's timestamptz encoder requires a datetime
    instance regardless of the ``::timestamptz`` cast on the bind. This adapter
    accepts a datetime, an ISO 8601 string, or ``None`` (now)."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # fromisoformat handles "+00:00" and naive ISO strings; assume UTC if naive.
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return datetime.now(timezone.utc)


_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {prefix}nodes (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector({dim}),
    utility_score DOUBLE PRECISION DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL,
    last_accessed TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{{}}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_{prefix}nodes_type ON {prefix}nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_{prefix}nodes_utility ON {prefix}nodes(utility_score);

CREATE TABLE IF NOT EXISTS {prefix}edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES {prefix}nodes(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES {prefix}nodes(id) ON DELETE CASCADE,
    relation_type TEXT NOT NULL,
    weight DOUBLE PRECISION DEFAULT 1.0,
    created_at TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{{}}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_{prefix}edges_source ON {prefix}edges(source_id);
CREATE INDEX IF NOT EXISTS idx_{prefix}edges_target ON {prefix}edges(target_id);
CREATE INDEX IF NOT EXISTS idx_{prefix}edges_type ON {prefix}edges(relation_type);
"""


class PostgresBackend:
    """Async storage backend using PostgreSQL + pgvector.

    Designed for multi-process deployments (web servers, worker pools)
    where embedded databases like Kuzu or SQLite can't safely share state.
    Uses asyncpg for native async I/O and pgvector for vector similarity.

    Usage:
        pool = await asyncpg.create_pool(dsn)
        backend = PostgresBackend(pool, embedding_dim=768)
        await backend.init()
        graph = PersistentGraph(backend=backend)
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        embedding_dim: int = 1536,
        table_prefix: str = "synap_",
    ) -> None:
        self._pool = pool
        self._dim = embedding_dim
        self._prefix = table_prefix
        self._nodes = f"{table_prefix}nodes"
        self._edges = f"{table_prefix}edges"

    async def init(self) -> None:
        """Create tables and indexes. Idempotent — safe to call on every startup."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                _SCHEMA_SQL.format(prefix=self._prefix, dim=self._dim)
            )

    # --- Node operations ---

    async def save_node(self, node: dict[str, Any]) -> None:
        embedding = node.get("embedding")
        embedding_str = _format_vector(embedding) if embedding else None

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._nodes}
                    (id, node_type, content, embedding, utility_score,
                     access_count, created_at, last_accessed, metadata)
                VALUES ($1, $2, $3, $4::vector, $5, $6, $7::timestamptz, $8::timestamptz, $9::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    node_type = EXCLUDED.node_type,
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    utility_score = EXCLUDED.utility_score,
                    access_count = EXCLUDED.access_count,
                    last_accessed = EXCLUDED.last_accessed,
                    metadata = EXCLUDED.metadata
                """,
                node["id"],
                node["node_type"],
                node["content"],
                embedding_str,
                float(node.get("utility_score", 1.0)),
                int(node.get("access_count", 0)),
                _coerce_timestamp(node.get("created_at")),
                _coerce_timestamp(node.get("last_accessed")),
                json.dumps(node.get("metadata", {})),
            )

    async def load_node(self, node_id: str) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                f"""
                SELECT id, node_type, content, embedding::text,
                       utility_score, access_count,
                       created_at::text, last_accessed::text, metadata
                FROM {self._nodes} WHERE id = $1
                """,
                node_id,
            )
        if row is None:
            return None
        return _row_to_node(row)

    # --- Edge operations ---

    async def save_edge(self, edge: dict[str, Any]) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._edges}
                    (id, source_id, target_id, relation_type, weight, created_at, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::timestamptz, $7::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    relation_type = EXCLUDED.relation_type,
                    weight = EXCLUDED.weight,
                    metadata = EXCLUDED.metadata
                """,
                edge["id"],
                edge["source_id"],
                edge["target_id"],
                edge["relation_type"],
                float(edge.get("weight", 1.0)),
                _coerce_timestamp(edge.get("created_at")),
                json.dumps(edge.get("metadata", {})),
            )

    async def load_edges(
        self, node_id: str, edge_type: str | None = None
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            if edge_type:
                rows = await conn.fetch(
                    f"""
                    SELECT id, source_id, target_id, relation_type, weight,
                           created_at::text, metadata
                    FROM {self._edges}
                    WHERE (source_id = $1 OR target_id = $1) AND relation_type = $2
                    """,
                    node_id, edge_type,
                )
            else:
                rows = await conn.fetch(
                    f"""
                    SELECT id, source_id, target_id, relation_type, weight,
                           created_at::text, metadata
                    FROM {self._edges}
                    WHERE source_id = $1 OR target_id = $1
                    """,
                    node_id,
                )
        return [_row_to_edge(r) for r in rows]

    # --- Query ---

    async def query_nodes(
        self,
        node_type: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        conditions = ["1=1"]
        params: list[Any] = []
        idx = 1

        if node_type:
            conditions.append(f"node_type = ${idx}")
            params.append(node_type)
            idx += 1

        if filters:
            conditions.append(f"metadata @> ${idx}::jsonb")
            params.append(json.dumps(filters))
            idx += 1

        conditions.append(f"LIMIT ${idx}")
        params.append(limit)

        where = " AND ".join(conditions[:-1])
        limit_clause = conditions[-1]

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT id, node_type, content, embedding::text,
                       utility_score, access_count,
                       created_at::text, last_accessed::text, metadata
                FROM {self._nodes}
                WHERE {where}
                ORDER BY utility_score DESC
                {limit_clause}
                """,
                *params,
            )
        return [_row_to_node(r) for r in rows]

    # --- Counts ---

    async def node_count(self, node_type: str | None = None) -> int:
        async with self._pool.acquire() as conn:
            if node_type:
                row = await conn.fetchrow(
                    f"SELECT count(*) FROM {self._nodes} WHERE node_type = $1",
                    node_type,
                )
            else:
                row = await conn.fetchrow(f"SELECT count(*) FROM {self._nodes}")
        return row[0]

    async def edge_count(self, relation_type: str | None = None) -> int:
        async with self._pool.acquire() as conn:
            if relation_type:
                row = await conn.fetchrow(
                    f"SELECT count(*) FROM {self._edges} WHERE relation_type = $1",
                    relation_type,
                )
            else:
                row = await conn.fetchrow(f"SELECT count(*) FROM {self._edges}")
        return row[0]

    # --- Delete ---

    async def delete_node(self, node_id: str) -> None:
        async with self._pool.acquire() as conn:
            # Edges deleted by ON DELETE CASCADE
            await conn.execute(
                f"DELETE FROM {self._nodes} WHERE id = $1", node_id
            )

    async def delete_edge(self, edge_id: str) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._edges} WHERE id = $1", edge_id
            )

    # --- Vector similarity search ---

    async def similarity_search(
        self,
        embedding: list[float],
        node_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        vec_str = _format_vector(embedding)
        async with self._pool.acquire() as conn:
            if node_type:
                rows = await conn.fetch(
                    f"""
                    SELECT id, node_type, content, embedding::text,
                           utility_score, access_count,
                           created_at::text, last_accessed::text, metadata,
                           1 - (embedding <=> $1::vector) AS similarity
                    FROM {self._nodes}
                    WHERE node_type = $2 AND embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                    """,
                    vec_str, node_type, limit,
                )
            else:
                rows = await conn.fetch(
                    f"""
                    SELECT id, node_type, content, embedding::text,
                           utility_score, access_count,
                           created_at::text, last_accessed::text, metadata,
                           1 - (embedding <=> $1::vector) AS similarity
                    FROM {self._nodes}
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    vec_str, limit,
                )
        return [_row_to_node(r) for r in rows]

    # --- Graph traversal via recursive CTE ---

    async def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> list[dict[str, Any]]:
        type_filter = ""
        params: list[Any] = [start_id, max_depth, max_nodes]

        if edge_types:
            type_filter = "AND e.relation_type = ANY($4)"
            params.append(edge_types)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                WITH RECURSIVE traversal AS (
                    SELECT
                        CASE WHEN e.source_id = $1 THEN e.target_id
                             ELSE e.source_id END AS node_id,
                        1 AS depth,
                        ARRAY[CASE WHEN e.source_id = $1 THEN e.target_id
                                   ELSE e.source_id END] AS path
                    FROM {self._edges} e
                    WHERE (e.source_id = $1 OR e.target_id = $1)
                        {type_filter}

                    UNION ALL

                    SELECT
                        CASE WHEN e.source_id = t.node_id THEN e.target_id
                             ELSE e.source_id END,
                        t.depth + 1,
                        t.path || CASE WHEN e.source_id = t.node_id THEN e.target_id
                                       ELSE e.source_id END
                    FROM traversal t
                    JOIN {self._edges} e
                        ON (e.source_id = t.node_id OR e.target_id = t.node_id)
                    WHERE t.depth < $2
                        AND CASE WHEN e.source_id = t.node_id THEN e.target_id
                                 ELSE e.source_id END <> ALL(t.path)
                        AND CASE WHEN e.source_id = t.node_id THEN e.target_id
                                 ELSE e.source_id END <> $1
                        {type_filter}
                )
                SELECT DISTINCT ON (n.id)
                    n.id, n.node_type, n.content, n.embedding::text,
                    n.utility_score, n.access_count,
                    n.created_at::text, n.last_accessed::text, n.metadata
                FROM traversal t
                JOIN {self._nodes} n ON n.id = t.node_id
                LIMIT $3
                """,
                *params,
            )
        return [_row_to_node(r) for r in rows]

    async def close(self) -> None:
        await self._pool.close()


# --- Helpers ---


def _format_vector(embedding: list[float]) -> str:
    """Format embedding as pgvector literal string."""
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def _parse_vector(val: str | None) -> list[float] | None:
    """Parse pgvector text representation back to list."""
    if val is None:
        return None
    # pgvector returns '[0.1,0.2,...]'
    return [float(x) for x in val.strip("[]").split(",") if x.strip()]


def _row_to_node(row: asyncpg.Record) -> dict[str, Any]:
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return {
        "id": row["id"],
        "node_type": row["node_type"],
        "content": row["content"],
        "embedding": _parse_vector(row["embedding"]) if "embedding" in row.keys() else None,
        "utility_score": row["utility_score"],
        "access_count": row["access_count"],
        "created_at": row["created_at"],
        "last_accessed": row["last_accessed"],
        "metadata": meta if isinstance(meta, dict) else {},
    }


def _row_to_edge(row: asyncpg.Record) -> dict[str, Any]:
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return {
        "id": row["id"],
        "source_id": row["source_id"],
        "target_id": row["target_id"],
        "relation_type": row["relation_type"],
        "weight": row["weight"],
        "created_at": row["created_at"],
        "metadata": meta if isinstance(meta, dict) else {},
    }
