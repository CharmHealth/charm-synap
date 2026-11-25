"""SQLite storage backend for persistent memory graphs."""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from synap._utils import cosine_similarity
from synap.tenancy import (
    ALL_TENANTS,
    TenantFilter,
    TenantMismatchError,
    tenant_matches,
)


def _tenant_condition(tenant: TenantFilter, params: list[Any]) -> str | None:
    """SQL predicate for a tenant filter, appending any bind to ``params``."""
    if tenant is ALL_TENANTS:
        return None
    if tenant is None:
        return "tenant IS NULL"
    params.append(tenant)
    return "tenant = ?"


def _with_tenant(row: sqlite3.Row) -> dict[str, Any]:
    """Node dict from a row, with tenant taken from its column.

    The column is authoritative: rows written before it existed have no
    tenant in their ``data`` blob.
    """
    data = json.loads(row["data"])
    data["tenant"] = row["tenant"]
    return data


def _synchronized(method):
    """Serialize a public backend method on the instance lock.

    PersistentGraph dispatches sync backends via ``asyncio.to_thread``, so calls
    arrive on arbitrary worker threads. The connection is opened with
    ``check_same_thread=False`` — safe because sqlite3 runs in serialized mode
    (threadsafety == 3) — and this lock keeps multi-statement transactions
    (write_batch, delete_nodes_batch) from being interleaved by a concurrent
    call, and reads from seeing another call's uncommitted rows on the shared
    connection. The lock is reentrant so a read method may call another
    (traverse -> load_edges/load_node) without deadlocking.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class SQLiteBackend:
    """Persistent storage using SQLite.

    Stores nodes and edges as JSON blobs with indexed columns
    for common query patterns. Embedding similarity search uses
    brute-force cosine similarity (adequate for <100K nodes;
    swap to a vector extension for larger stores).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        # Reentrant so read methods can nest (traverse -> load_*); guards the
        # single connection shared across executor threads (see _synchronized).
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                utility_score REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                data TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
            CREATE INDEX IF NOT EXISTS idx_nodes_utility ON nodes(utility_score);

            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                data TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES nodes(id),
                FOREIGN KEY (target_id) REFERENCES nodes(id)
            );

            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
            CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(relation_type);
        """)
        # Forward-migrate a store created before the tenant column existed
        # (see synap.tenancy). SQLite has no ADD COLUMN IF NOT EXISTS.
        try:
            self._conn.execute("ALTER TABLE nodes ADD COLUMN tenant TEXT")
        except sqlite3.OperationalError:
            pass  # column already present
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nodes_tenant_type ON nodes(tenant, node_type)"
        )
        self._conn.commit()

    def _write_node_row(self, node: dict[str, Any]) -> None:
        """Execute a node upsert without committing (for batching)."""
        embedding_json = (
            json.dumps(node.get("embedding"))
            if node.get("embedding") is not None
            else None
        )
        tenant = node.get("tenant")
        # INSERT OR REPLACE keys on id alone; an id that exists under another
        # tenant would be overwritten and moved. Refuse it instead.
        existing = self._conn.execute(
            "SELECT tenant FROM nodes WHERE id = ?", (node["id"],)
        ).fetchone()
        if existing is not None and existing["tenant"] != tenant:
            raise TenantMismatchError(
                f"node {node['id']!r} belongs to a different tenant; "
                "refusing to overwrite it"
            )
        self._conn.execute(
            """INSERT OR REPLACE INTO nodes
               (id, node_type, content, embedding, utility_score,
                access_count, created_at, last_accessed, metadata, data, tenant)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                node["id"],
                node["node_type"],
                node["content"],
                embedding_json,
                node.get("utility_score", 1.0),
                node.get("access_count", 0),
                node["created_at"],
                node["last_accessed"],
                json.dumps(node.get("metadata", {})),
                json.dumps(node),
                tenant,
            ),
        )

    @_synchronized
    def save_node(self, node: dict[str, Any]) -> None:
        self._write_node_row(node)
        self._conn.commit()

    def _write_edge_row(self, edge: dict[str, Any]) -> None:
        """Execute an edge upsert without committing (for batching)."""
        self._conn.execute(
            """INSERT OR REPLACE INTO edges
               (id, source_id, target_id, relation_type, weight,
                created_at, metadata, data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                edge["id"],
                edge["source_id"],
                edge["target_id"],
                edge["relation_type"],
                edge.get("weight", 1.0),
                edge["created_at"],
                json.dumps(edge.get("metadata", {})),
                json.dumps(edge),
            ),
        )

    @_synchronized
    def save_edge(self, edge: dict[str, Any]) -> None:
        self._write_edge_row(edge)
        self._conn.commit()

    @_synchronized
    def write_batch(
        self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
    ) -> None:
        """Write nodes then edges atomically, committing once; roll back on error."""
        try:
            for node in nodes:
                self._write_node_row(node)
            for edge in edges:
                self._write_edge_row(edge)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    @_synchronized
    def load_node(self, node_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data, tenant FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return _with_tenant(row)

    @_synchronized
    def load_edges(
        self, node_id: str, edge_type: str | None = None
    ) -> list[dict[str, Any]]:
        if edge_type:
            rows = self._conn.execute(
                """SELECT data FROM edges
                   WHERE (source_id = ? OR target_id = ?) AND relation_type = ?""",
                (node_id, node_id, edge_type),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT data FROM edges WHERE source_id = ? OR target_id = ?",
                (node_id, node_id),
            ).fetchall()
        return [json.loads(r["data"]) for r in rows]

    @_synchronized
    def query_nodes(
        self,
        node_type: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        *,
        tenant: TenantFilter = ALL_TENANTS,
    ) -> list[dict[str, Any]]:
        query = "SELECT data, metadata, tenant FROM nodes WHERE 1=1"
        params: list[Any] = []

        if node_type:
            query += " AND node_type = ?"
            params.append(node_type)

        tenant_cond = _tenant_condition(tenant, params)
        if tenant_cond:
            query += f" AND {tenant_cond}"

        query += " ORDER BY utility_score DESC"

        # When metadata filters are present the LIMIT must apply *after*
        # filtering (parity with the Postgres pushdown), so it is enforced in
        # the Python loop below instead of in SQL. Without filters SQL limits
        # directly.
        if not filters:
            query += " LIMIT ?"
            params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        results = []
        for row in rows:
            data = _with_tenant(row)
            if filters:
                meta = json.loads(row["metadata"])
                if all(meta.get(k) == v for k, v in filters.items()):
                    results.append(data)
                    if len(results) >= limit:
                        break
            else:
                results.append(data)
        return results

    @_synchronized
    def list_tenants(self) -> list[str]:
        """Every tenant that owns at least one node (untenanted excluded)."""
        rows = self._conn.execute(
            "SELECT DISTINCT tenant FROM nodes WHERE tenant IS NOT NULL ORDER BY tenant"
        ).fetchall()
        return [r[0] for r in rows]

    @_synchronized
    def node_count(
        self, node_type: str | None = None, *, tenant: TenantFilter = ALL_TENANTS
    ) -> int:
        query = "SELECT COUNT(*) FROM nodes WHERE 1=1"
        params: list[Any] = []
        if node_type:
            query += " AND node_type = ?"
            params.append(node_type)
        tenant_cond = _tenant_condition(tenant, params)
        if tenant_cond:
            query += f" AND {tenant_cond}"
        return self._conn.execute(query, params).fetchone()[0]

    @_synchronized
    def edge_count(
        self,
        relation_type: str | None = None,
        *,
        tenant: TenantFilter = ALL_TENANTS,
    ) -> int:
        # An edge belongs to its source node's tenant; edges never join two
        # tenants (the graph layer refuses that write).
        query = "SELECT COUNT(*) FROM edges e"
        params: list[Any] = []
        conditions: list[str] = []
        tenant_cond = _tenant_condition(tenant, params)
        if tenant_cond:
            query += " JOIN nodes n ON n.id = e.source_id"
            conditions.append(tenant_cond.replace("tenant", "n.tenant", 1))
        if relation_type:
            conditions.append("e.relation_type = ?")
            params.append(relation_type)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        return self._conn.execute(query, params).fetchone()[0]

    def _delete_node_rows(self, node_id: str) -> None:
        """Delete a node and its edges without committing (for batching)."""
        self._conn.execute(
            "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
            (node_id, node_id),
        )
        self._conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))

    @_synchronized
    def delete_node(self, node_id: str) -> None:
        self._delete_node_rows(node_id)
        self._conn.commit()

    @_synchronized
    def delete_nodes_batch(self, node_ids: list[str]) -> None:
        """Delete multiple nodes (and their edges) atomically, one commit."""
        try:
            for node_id in node_ids:
                self._delete_node_rows(node_id)
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    @_synchronized
    def delete_edge(self, edge_id: str) -> None:
        self._conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
        self._conn.commit()

    @_synchronized
    def similarity_search(
        self,
        embedding: list[float],
        node_type: str | None = None,
        limit: int = 10,
        *,
        tenant: TenantFilter = ALL_TENANTS,
    ) -> list[dict[str, Any]]:
        """Brute-force cosine similarity search."""
        query = "SELECT data, embedding, tenant FROM nodes WHERE embedding IS NOT NULL"
        params: list[Any] = []
        if node_type:
            query += " AND node_type = ?"
            params.append(node_type)
        tenant_cond = _tenant_condition(tenant, params)
        if tenant_cond:
            query += f" AND {tenant_cond}"

        rows = self._conn.execute(query, params).fetchall()
        scored: list[tuple[float, dict[str, Any]]] = []

        for row in rows:
            node_embedding = json.loads(row["embedding"])
            if len(node_embedding) != len(embedding):
                data = json.loads(row["data"])
                raise ValueError(
                    f"embedding dimension mismatch in similarity search: query has "
                    f"{len(embedding)}, node {data.get('id')} has {len(node_embedding)}; "
                    f"the store holds inconsistent embedding dimensions"
                )
            sim = cosine_similarity(embedding, node_embedding)
            scored.append((sim, _with_tenant(row)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [data for _, data in scored[:limit]]

    @_synchronized
    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_depth: int = 2,
        max_nodes: int = 50,
        *,
        tenant: TenantFilter = ALL_TENANTS,
    ) -> list[dict[str, Any]]:
        """BFS traversal from start node. Not graph-native — walks edges in Python.

        A node outside ``tenant`` is neither returned nor walked through.
        """
        visited: set[str] = {start_id}
        frontier = [start_id]
        results: list[dict[str, Any]] = []

        for _ in range(max_depth):
            next_frontier: list[str] = []
            for nid in frontier:
                edges = self.load_edges(nid)
                for edge in edges:
                    if edge_types and edge["relation_type"] not in edge_types:
                        continue
                    neighbor_id = (
                        edge["target_id"] if edge["source_id"] == nid else edge["source_id"]
                    )
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        node = self.load_node(neighbor_id)
                        if node and tenant_matches(node.get("tenant"), tenant):
                            results.append(node)
                            if len(results) >= max_nodes:
                                return results
                            next_frontier.append(neighbor_id)
            frontier = next_frontier
            if not frontier:
                break

        return results

    @_synchronized
    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
