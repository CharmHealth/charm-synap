"""PostgreSQL storage backend — async, multi-process safe, pgvector similarity."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from psycopg import AsyncCursor, errors
from psycopg.rows import DictRow, dict_row
from psycopg_pool import AsyncConnectionPool


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _coerce_timestamp(value: Any) -> datetime:
    """Coerce a caller-supplied timestamp into a timezone-aware datetime.

    Synap's higher layers serialize timestamps as ISO strings so they round-trip
    through any backend. Binding a datetime rather than the string keeps a naive
    ISO string from being read in the server session's time zone: it is pinned
    to UTC here. This adapter accepts a datetime, an ISO 8601 string, or
    ``None`` (now)."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        # fromisoformat handles "+00:00" and naive ISO strings; assume UTC if naive.
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    return datetime.now(UTC)


def _coerce_timestamp_optional(value: Any) -> datetime | None:
    """Like ``_coerce_timestamp`` but preserves ``None``.

    Validity-window fields (``valid_from``/``valid_until``) are nullable — a
    missing bound must stay NULL rather than defaulting to now()."""
    if value is None:
        return None
    return _coerce_timestamp(value)


# Separate from _SCHEMA_SQL so a privilege failure here can be reported for what
# it is. On managed Postgres the application role is not a superuser and cannot
# create extensions; init() turns that into an actionable message rather than a
# bare "permission denied" from somewhere inside startup.
_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector;"

# pgvector's hnsw index accepts at most 2000 dimensions for the `vector` type.
# Checked at construction rather than at index-creation time because otherwise
# the failure lands a long way from its cause: a 3072-dimension model (OpenAI's
# text-embedding-3-large, for one) would get a raw driver error out of init()
# and the backend would never start, with nothing naming the dimension as the
# problem.
#
# This is a new ceiling. Before the hnsw index existed, any dimension worked
# because nothing indexed the column. Refusing here rather than skipping the
# index is deliberate: a store that silently has no similarity index is the
# exact failure this index was added to fix, so it is better to refuse the
# configuration than to honour it without the index.
HNSW_MAX_DIM = 2000

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS {prefix}nodes (
    id TEXT PRIMARY KEY,
    node_type TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding vector({dim}),
    utility_score DOUBLE PRECISION DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL,
    last_accessed TIMESTAMPTZ NOT NULL,
    metadata JSONB DEFAULT '{{}}'::jsonb,
    valid_from TIMESTAMPTZ,
    valid_until TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_{prefix}nodes_type ON {prefix}nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_{prefix}nodes_utility ON {prefix}nodes(utility_score);

-- The index the similarity search actually needs. search_similar orders by
-- `embedding <=> <query vector>` (cosine distance), and without an index for that
-- operator every search is a sequential scan computing a distance for every
-- node in the graph — which is exactly the query the graph exists to serve.
--
-- vector_cosine_ops because the query uses <=>. An l2 index cannot answer it:
-- the planner will not use an operator class that does not match the operator,
-- so a mismatch here fails open as a full scan and nothing errors.
--
-- hnsw rather than ivfflat because a new deployment starts empty. ivfflat
-- builds its lists by clustering the rows present at build time, so an index
-- created against an empty table has nothing to cluster and stays useless
-- until someone rebuilds it after data arrives. hnsw builds incrementally and
-- is correct from zero rows, at the cost of more memory and slower inserts.
-- Bounded by pgvector's 2000-dimension limit for hnsw; Titan Embed v2 is 1024.
CREATE INDEX IF NOT EXISTS idx_{prefix}nodes_embedding
    ON {prefix}nodes USING hnsw (embedding vector_cosine_ops);

-- Forward-migrate a store created before these columns existed (idempotent).
-- Additive-only; a versioned migration path is ROADMAP task 7.
ALTER TABLE {prefix}nodes ADD COLUMN IF NOT EXISTS valid_from TIMESTAMPTZ;
ALTER TABLE {prefix}nodes ADD COLUMN IF NOT EXISTS valid_until TIMESTAMPTZ;

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


# Columns init() checks for when create_schema=False. Kept next to _SCHEMA_SQL so
# a column added there is added here too; a store missing one fails at startup
# with the column named, not on the first query that touches it.
_NODE_COLUMNS = frozenset({
    "id", "node_type", "content", "embedding", "utility_score", "access_count",
    "created_at", "last_accessed", "metadata", "valid_from", "valid_until",
})
_EDGE_COLUMNS = frozenset({
    "id", "source_id", "target_id", "relation_type", "weight", "created_at",
    "metadata",
})


def schema_sql(table_prefix: str = "synap_", embedding_dim: int = 1536) -> str:
    """The DDL ``init()`` runs when ``create_schema=True``.

    For deployments where the application role may not run DDL: put this in
    the schema migrations (Flyway, Alembic, ...), run as a role that may, and
    construct the backend with ``create_schema=False``. Does not include
    ``CREATE EXTENSION vector``, which needs a superuser on managed Postgres.
    """
    return _SCHEMA_SQL.format(prefix=table_prefix, dim=embedding_dim)


class PostgresBackend:
    """Async storage backend using PostgreSQL + pgvector.

    Designed for multi-process deployments (web servers, worker pools)
    where embedded databases like Kuzu or SQLite can't safely share state.
    Uses psycopg 3 (libpq) for async I/O and pgvector for vector similarity.

    Usage:
        pool = AsyncConnectionPool(dsn, open=False)
        await pool.open()
        backend = PostgresBackend(pool, embedding_dim=768)
        await backend.init()
        graph = PersistentGraph(backend=backend)

    The pool may be shared with the rest of the application, whatever its
    settings: every query here asks for dict rows on its own cursor and runs
    unprepared, and writes are committed whether or not the pool's
    connections are in autocommit mode.

    Why psycopg and not asyncpg: some Postgres proxies, ProxySQL 3.0 among
    them, don't implement the extended protocol's Flush message, which asyncpg
    sends on every query. libpq sends Sync instead. For the same reason this
    backend never uses ``executemany()`` or pipeline mode, both of which send
    Flush. Statements are never prepared because a proxy that shares backend
    connections may run the next query where the prepared statement doesn't
    exist.

    ``create_schema=False`` is for deployments where the application role
    can't run DDL: the tables come from the deployment's own migrations (see
    :func:`schema_sql`), and ``init()`` only checks that they are there, with
    the columns and the embedding dimension this backend expects.
    """

    def __init__(
        self,
        pool: AsyncConnectionPool,
        embedding_dim: int = 1536,
        table_prefix: str = "synap_",
        create_schema: bool = True,
    ) -> None:
        if embedding_dim > HNSW_MAX_DIM:
            raise ValueError(
                f"embedding_dim={embedding_dim} exceeds pgvector's hnsw limit "
                f"of {HNSW_MAX_DIM} dimensions, so the similarity index this "
                "backend creates cannot be built. Use a model with fewer "
                "dimensions, or reduce the vector before storing it."
            )
        self._pool = pool
        self._dim = embedding_dim
        self._prefix = table_prefix
        self._create_schema = create_schema
        self._nodes = f"{table_prefix}nodes"
        self._edges = f"{table_prefix}edges"

    @asynccontextmanager
    async def _cursor(self) -> AsyncIterator[AsyncCursor[DictRow]]:
        """A dict-row cursor on a pooled connection, for one operation."""
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                yield cur

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[AsyncCursor[DictRow]]:
        """Like ``_cursor``, inside one transaction (rolled back on error)."""
        async with self._pool.connection() as conn, conn.transaction():
            async with conn.cursor(row_factory=dict_row) as cur:
                yield cur

    async def init(self) -> None:
        """Create the tables and indexes, or check them when create_schema=False.

        Idempotent — safe to call on every startup."""
        if self._create_schema:
            await self._create()
        else:
            await self._check_schema()

    async def _create(self) -> None:
        async with self._cursor() as cur:
            try:
                await cur.execute(_EXTENSION_SQL, prepare=False)
            except errors.InsufficientPrivilege as exc:
                # Verified against Postgres 14: a non-superuser running
                # `CREATE EXTENSION IF NOT EXISTS vector` SUCCEEDS when the
                # extension is already installed (NOTICE, skipping) and fails
                # with 42501 only when it has to create it. So this is a
                # one-time provisioning step, not an ongoing privilege the
                # application role needs.
                raise RuntimeError(
                    "pgvector is not installed in this database and this role "
                    "cannot install it. An administrator must run "
                    "'CREATE EXTENSION vector;' against this database once, "
                    "before synap first starts — on RDS or Aurora that means a "
                    "role with rds_superuser. The application role needs no "
                    "extension privileges once it exists."
                ) from exc
            except errors.UndefinedFile as exc:
                # 58P01. Distinct from the privilege case above and much more
                # common: the extension's control file is not on the server at
                # all, so no role can create it and the advice to find an
                # administrator is wrong. A plain `postgres` image has no
                # pgvector; RDS, Aurora and pgvector/pgvector all ship it.
                raise RuntimeError(
                    "pgvector is not installed on this Postgres server — its "
                    "extension control file is missing, so no role can create "
                    "it and this is not a permissions problem. Install the "
                    "server package (postgresql-<version>-pgvector), or use an "
                    "image that ships it: pgvector/pgvector, or RDS/Aurora."
                ) from exc
            try:
                # No parameters, so psycopg sends this as one simple query and
                # the several statements in it run together.
                await cur.execute(
                    schema_sql(self._prefix, self._dim), prepare=False
                )
            except errors.UndefinedObject as exc:
                # 42704. In this statement the only object that can be
                # undefined is the hnsw access method: the `vector` type itself
                # would have failed at CREATE EXTENSION above, and everything
                # else here is built-in. hnsw arrived in pgvector 0.5.0, so an
                # older extension reaches this line and fails on the index.
                raise RuntimeError(
                    "This server's pgvector is too old to build the similarity "
                    "index — the hnsw access method was added in pgvector "
                    "0.5.0. Run 'ALTER EXTENSION vector UPDATE;', or upgrade "
                    "the server package if the installed version is already "
                    "the newest available."
                ) from exc

    async def _check_schema(self) -> None:
        """Refuse to start against tables that aren't the ones this backend uses."""
        async with self._cursor() as cur:
            await cur.execute(
                """
                SELECT c.relname AS table_name, a.attname AS column_name,
                       format_type(a.atttypid, a.atttypmod) AS column_type
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                WHERE a.attrelid IN (to_regclass(%(nodes)s), to_regclass(%(edges)s))
                  AND a.attnum > 0 AND NOT a.attisdropped
                """,
                {"nodes": self._nodes, "edges": self._edges},
                prepare=False,
            )
            rows = await cur.fetchall()
        columns: dict[str, dict[str, str]] = {self._nodes: {}, self._edges: {}}
        for row in rows:
            columns[row["table_name"]][row["column_name"]] = row["column_type"]

        problems = []
        for table, expected in ((self._nodes, _NODE_COLUMNS), (self._edges, _EDGE_COLUMNS)):
            if not columns[table]:
                problems.append(f"table {table} does not exist")
                continue
            missing = sorted(expected - columns[table].keys())
            if missing:
                problems.append(f"{table} has no column {', '.join(missing)}")
        embedding_type = columns[self._nodes].get("embedding")
        if embedding_type and embedding_type != f"vector({self._dim})":
            problems.append(
                f"{self._nodes}.embedding is {embedding_type}, but this backend "
                f"was configured with embedding_dim={self._dim}"
            )
        if problems:
            raise RuntimeError(
                "synap's Postgres tables don't match this backend (create_schema="
                "False, so they must come from the deployment's migrations; "
                "synap.backends.postgres.schema_sql() gives the DDL): "
                + "; ".join(problems)
            )

    # --- Node operations ---

    async def _write_node(self, cur: AsyncCursor[DictRow], node: dict[str, Any]) -> None:
        """Upsert one node with the given cursor (for save_node and batches)."""
        embedding = node.get("embedding")
        embedding_str = _format_vector(embedding) if embedding else None
        await cur.execute(
            f"""
            INSERT INTO {self._nodes}
                (id, node_type, content, embedding, utility_score,
                 access_count, created_at, last_accessed, metadata,
                 valid_from, valid_until)
            VALUES (%s, %s, %s, %s::vector, %s, %s, %s::timestamptz, %s::timestamptz, %s::jsonb,
                    %s::timestamptz, %s::timestamptz)
            ON CONFLICT (id) DO UPDATE SET
                node_type = EXCLUDED.node_type,
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                utility_score = EXCLUDED.utility_score,
                access_count = EXCLUDED.access_count,
                last_accessed = EXCLUDED.last_accessed,
                metadata = EXCLUDED.metadata,
                valid_from = EXCLUDED.valid_from,
                valid_until = EXCLUDED.valid_until
            """,
            (
                node["id"],
                node["node_type"],
                node["content"],
                embedding_str,
                float(node.get("utility_score", 1.0)),
                int(node.get("access_count", 0)),
                _coerce_timestamp(node.get("created_at")),
                _coerce_timestamp(node.get("last_accessed")),
                json.dumps(node.get("metadata", {})),
                _coerce_timestamp_optional(node.get("valid_from")),
                _coerce_timestamp_optional(node.get("valid_until")),
            ),
            prepare=False,
        )

    async def save_node(self, node: dict[str, Any]) -> None:
        async with self._transaction() as cur:
            await self._write_node(cur, node)

    async def load_node(self, node_id: str) -> dict[str, Any] | None:
        async with self._cursor() as cur:
            await cur.execute(
                f"""
                SELECT id, node_type, content, embedding::text AS embedding,
                       utility_score, access_count,
                       created_at::text AS created_at,
                       last_accessed::text AS last_accessed, metadata,
                       valid_from::text AS valid_from,
                       valid_until::text AS valid_until
                FROM {self._nodes} WHERE id = %s
                """,
                (node_id,),
                prepare=False,
            )
            row = await cur.fetchone()
        if row is None:
            return None
        return _row_to_node(row)

    # --- Edge operations ---

    async def _write_edge(self, cur: AsyncCursor[DictRow], edge: dict[str, Any]) -> None:
        """Upsert one edge with the given cursor (for save_edge and batches)."""
        await cur.execute(
            f"""
            INSERT INTO {self._edges}
                (id, source_id, target_id, relation_type, weight, created_at, metadata)
            VALUES (%s, %s, %s, %s, %s, %s::timestamptz, %s::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                relation_type = EXCLUDED.relation_type,
                weight = EXCLUDED.weight,
                metadata = EXCLUDED.metadata
            """,
            (
                edge["id"],
                edge["source_id"],
                edge["target_id"],
                edge["relation_type"],
                float(edge.get("weight", 1.0)),
                _coerce_timestamp(edge.get("created_at")),
                json.dumps(edge.get("metadata", {})),
            ),
            prepare=False,
        )

    async def save_edge(self, edge: dict[str, Any]) -> None:
        async with self._transaction() as cur:
            await self._write_edge(cur, edge)

    async def write_batch(
        self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
    ) -> None:
        """Write nodes then edges atomically in one transaction; rolls back on error.

        One statement per row, not executemany(): see the class docstring."""
        async with self._transaction() as cur:
            for node in nodes:
                await self._write_node(cur, node)
            for edge in edges:
                await self._write_edge(cur, edge)

    async def load_edges(
        self, node_id: str, edge_type: str | None = None
    ) -> list[dict[str, Any]]:
        async with self._cursor() as cur:
            if edge_type:
                await cur.execute(
                    f"""
                    SELECT id, source_id, target_id, relation_type, weight,
                           created_at::text AS created_at, metadata
                    FROM {self._edges}
                    WHERE (source_id = %(node_id)s OR target_id = %(node_id)s)
                      AND relation_type = %(edge_type)s
                    """,
                    {"node_id": node_id, "edge_type": edge_type},
                    prepare=False,
                )
            else:
                await cur.execute(
                    f"""
                    SELECT id, source_id, target_id, relation_type, weight,
                           created_at::text AS created_at, metadata
                    FROM {self._edges}
                    WHERE source_id = %(node_id)s OR target_id = %(node_id)s
                    """,
                    {"node_id": node_id},
                    prepare=False,
                )
            rows = await cur.fetchall()
        return [_row_to_edge(r) for r in rows]

    # --- Query ---

    async def query_nodes(
        self,
        node_type: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        conditions = ["1=1"]
        params: dict[str, Any] = {"limit": limit}

        if node_type:
            conditions.append("node_type = %(node_type)s")
            params["node_type"] = node_type

        if filters:
            conditions.append("metadata @> %(filters)s::jsonb")
            params["filters"] = json.dumps(filters)

        where = " AND ".join(conditions)

        async with self._cursor() as cur:
            await cur.execute(
                f"""
                SELECT id, node_type, content, embedding::text AS embedding,
                       utility_score, access_count,
                       created_at::text AS created_at,
                       last_accessed::text AS last_accessed, metadata,
                       valid_from::text AS valid_from,
                       valid_until::text AS valid_until
                FROM {self._nodes}
                WHERE {where}
                ORDER BY utility_score DESC
                LIMIT %(limit)s
                """,
                params,
                prepare=False,
            )
            rows = await cur.fetchall()
        return [_row_to_node(r) for r in rows]

    # --- Counts ---

    async def node_count(self, node_type: str | None = None) -> int:
        async with self._cursor() as cur:
            if node_type:
                await cur.execute(
                    f"SELECT count(*) AS n FROM {self._nodes} WHERE node_type = %s",
                    (node_type,),
                    prepare=False,
                )
            else:
                await cur.execute(
                    f"SELECT count(*) AS n FROM {self._nodes}", prepare=False
                )
            row = await cur.fetchone()
        assert row is not None  # count(*) always returns one row
        return int(row["n"])

    async def edge_count(self, relation_type: str | None = None) -> int:
        async with self._cursor() as cur:
            if relation_type:
                await cur.execute(
                    f"SELECT count(*) AS n FROM {self._edges} WHERE relation_type = %s",
                    (relation_type,),
                    prepare=False,
                )
            else:
                await cur.execute(
                    f"SELECT count(*) AS n FROM {self._edges}", prepare=False
                )
            row = await cur.fetchone()
        assert row is not None  # count(*) always returns one row
        return int(row["n"])

    # --- Delete ---

    async def delete_node(self, node_id: str) -> None:
        async with self._transaction() as cur:
            # Edges deleted by ON DELETE CASCADE
            await cur.execute(
                f"DELETE FROM {self._nodes} WHERE id = %s", (node_id,), prepare=False
            )

    async def delete_nodes_batch(self, node_ids: list[str]) -> None:
        """Delete multiple nodes (edges cascade) atomically in one statement.

        Used to evict a whole episode all-or-nothing.
        """
        if not node_ids:
            return
        async with self._transaction() as cur:
            await cur.execute(
                f"DELETE FROM {self._nodes} WHERE id = ANY(%s::text[])",
                (list(node_ids),),
                prepare=False,
            )

    async def delete_edge(self, edge_id: str) -> None:
        async with self._transaction() as cur:
            await cur.execute(
                f"DELETE FROM {self._edges} WHERE id = %s", (edge_id,), prepare=False
            )

    # --- Vector similarity search ---

    async def similarity_search(
        self,
        embedding: list[float],
        node_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"vec": _format_vector(embedding), "limit": limit}
        type_filter = ""
        if node_type:
            type_filter = "node_type = %(node_type)s AND"
            params["node_type"] = node_type
        async with self._cursor() as cur:
            await cur.execute(
                f"""
                SELECT id, node_type, content, embedding::text AS embedding,
                       utility_score, access_count,
                       created_at::text AS created_at,
                       last_accessed::text AS last_accessed, metadata,
                       valid_from::text AS valid_from,
                       valid_until::text AS valid_until,
                       1 - (embedding <=> %(vec)s::vector) AS similarity
                FROM {self._nodes}
                WHERE {type_filter} embedding IS NOT NULL
                ORDER BY embedding <=> %(vec)s::vector
                LIMIT %(limit)s
                """,
                params,
                prepare=False,
            )
            rows = await cur.fetchall()
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
        params: dict[str, Any] = {
            "start": start_id, "max_depth": max_depth, "max_nodes": max_nodes,
        }

        if edge_types:
            type_filter = "AND e.relation_type = ANY(%(edge_types)s::text[])"
            params["edge_types"] = list(edge_types)

        async with self._cursor() as cur:
            await cur.execute(
                f"""
                WITH RECURSIVE traversal AS (
                    SELECT
                        CASE WHEN e.source_id = %(start)s THEN e.target_id
                             ELSE e.source_id END AS node_id,
                        1 AS depth,
                        ARRAY[CASE WHEN e.source_id = %(start)s THEN e.target_id
                                   ELSE e.source_id END] AS path
                    FROM {self._edges} e
                    WHERE (e.source_id = %(start)s OR e.target_id = %(start)s)
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
                    WHERE t.depth < %(max_depth)s
                        AND CASE WHEN e.source_id = t.node_id THEN e.target_id
                                 ELSE e.source_id END <> ALL(t.path)
                        AND CASE WHEN e.source_id = t.node_id THEN e.target_id
                                 ELSE e.source_id END <> %(start)s
                        {type_filter}
                )
                SELECT DISTINCT ON (n.id)
                    n.id, n.node_type, n.content, n.embedding::text AS embedding,
                    n.utility_score, n.access_count,
                    n.created_at::text AS created_at,
                    n.last_accessed::text AS last_accessed, n.metadata,
                    n.valid_from::text AS valid_from,
                    n.valid_until::text AS valid_until
                FROM traversal t
                JOIN {self._nodes} n ON n.id = t.node_id
                LIMIT %(max_nodes)s
                """,
                params,
                prepare=False,
            )
            rows = await cur.fetchall()
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


def _row_to_node(row: dict[str, Any]) -> dict[str, Any]:
    meta = row["metadata"]
    if isinstance(meta, str):
        meta = json.loads(meta)
    return {
        "id": row["id"],
        "node_type": row["node_type"],
        "content": row["content"],
        "embedding": _parse_vector(row.get("embedding")),
        "utility_score": row["utility_score"],
        "access_count": row["access_count"],
        "created_at": row["created_at"],
        "last_accessed": row["last_accessed"],
        "metadata": meta if isinstance(meta, dict) else {},
        "valid_from": row.get("valid_from"),
        "valid_until": row.get("valid_until"),
    }


def _row_to_edge(row: dict[str, Any]) -> dict[str, Any]:
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
