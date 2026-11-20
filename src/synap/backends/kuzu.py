"""Kùzu graph database backend — native graph traversal + vector search."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import kuzu

from synap.tenancy import ALL_TENANTS, TenantFilter, TenantMismatchError

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE NODE TABLE IF NOT EXISTS MemoryNode(
    id STRING,
    node_type STRING,
    content STRING,
    embedding DOUBLE[{dim}],
    utility_score DOUBLE DEFAULT 1.0,
    access_count INT64 DEFAULT 0,
    created_at STRING,
    last_accessed STRING,
    metadata STRING DEFAULT '{{}}',
    valid_from STRING,
    valid_until STRING,
    PRIMARY KEY(id)
);

CREATE REL TABLE IF NOT EXISTS MemoryEdge(
    FROM MemoryNode TO MemoryNode,
    id STRING,
    relation_type STRING,
    weight DOUBLE DEFAULT 1.0,
    created_at STRING,
    metadata STRING DEFAULT '{{}}'
);
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _node_return(var: str = "n") -> str:
    """RETURN list for a node, in _row_to_node's column order."""
    return (
        f"{var}.id, {var}.node_type, {var}.content, {var}.embedding, "
        f"{var}.utility_score, {var}.access_count, "
        f"{var}.created_at, {var}.last_accessed, {var}.metadata, "
        f"{var}.valid_from, {var}.valid_until, {var}.tenant"
    )


def _tenant_condition(
    var: str, tenant: TenantFilter, params: dict[str, Any]
) -> str | None:
    """Cypher predicate for a tenant filter, adding any bind to ``params``."""
    if tenant is ALL_TENANTS:
        return None
    if tenant is None:
        return f"{var}.tenant IS NULL"
    params["tenant"] = tenant
    return f"{var}.tenant = $tenant"


def _where(conditions: list[str]) -> str:
    return f"WHERE {' AND '.join(conditions)}" if conditions else ""


class KuzuBackend:
    """Graph-native storage backend using Kùzu.

    Provides native graph traversal via Cypher, native vector
    similarity via array_cosine_similarity, and file-based
    persistence with zero server infrastructure.

    Thread-safe: each operation creates its own connection from the
    shared Database object. Kuzu databases are thread-safe but
    connections are not.
    """

    def __init__(
        self,
        path: str | Path,
        embedding_dim: int,
        buffer_pool_mb: int = 256,
    ) -> None:
        self._path = str(path)
        self._embedding_dim = embedding_dim
        self._db: kuzu.Database | None = kuzu.Database(
            self._path, buffer_pool_size=buffer_pool_mb * 1024 * 1024
        )
        self._ensure_schema()

    def _conn(self) -> kuzu.Connection:
        """Create a fresh connection for the current operation."""
        assert self._db is not None  # backend is closed
        return kuzu.Connection(self._db)

    def _ensure_schema(self) -> None:
        """Idempotent schema creation.

        On reopen, the embedding-column dimension is fixed at CREATE time, so a
        caller who opens an existing store with a different ``embedding_dim``
        than it was created with would silently write vectors the schema can't
        hold. Validate the persisted dimension and fail loud instead.
        """
        conn = self._conn()
        existing_dim = self._persisted_embedding_dim(conn)
        if existing_dim is not None and existing_dim != self._embedding_dim:
            raise ValueError(
                f"Kuzu store at {self._path} was created with embedding_dim="
                f"{existing_dim}, but embedding_dim={self._embedding_dim} was "
                f"requested; the on-disk vector column cannot be resized"
            )
        for stmt in _SCHEMA_SQL.format(dim=self._embedding_dim).split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    conn.execute(stmt)
                except RuntimeError:
                    pass  # Table already exists

        # Forward-migrate a store created before a column existed. Kuzu's
        # ALTER ... ADD has no IF NOT EXISTS, so swallow the "already exists"
        # error. Additive-only; a versioned migration path is ROADMAP task 7.
        for column in ("valid_from", "valid_until", "tenant"):
            try:
                conn.execute(f"ALTER TABLE MemoryNode ADD {column} STRING")
            except RuntimeError:
                pass  # column already present

    def _persisted_embedding_dim(self, conn: kuzu.Connection) -> int | None:
        """The embedding-column dimension of an existing MemoryNode table, if any.

        Returns None when the table does not exist yet (fresh database).
        """
        try:
            result = conn.execute("CALL TABLE_INFO('MemoryNode') RETURN *")
        except RuntimeError:
            return None  # table not created yet
        for row in self._collect_rows(result):
            cells = [c for c in row if isinstance(c, str)]
            if not any(c == "embedding" for c in cells):
                continue
            for c in cells:
                m = re.search(r"\[(\d+)\]", c)
                if m:
                    return int(m.group(1))
        return None

    # --- Node operations ---

    def _write_node(self, conn: kuzu.Connection, node: dict[str, Any]) -> None:
        """Upsert one node on the given connection (dim-checked embedding)."""
        embedding = node.get("embedding")
        embedding_val = self._format_embedding(embedding) if embedding else None
        tenant = node.get("tenant")
        # MERGE matches on id alone, so an id that already exists under another
        # tenant would be overwritten and moved. Refuse it instead.
        #
        # This precheck and the MERGE below are two separate statements, so
        # there is a TOCTOU race: another connection can CREATE the same id
        # under a different tenant between them, and this MERGE would then
        # hit ON MATCH (silently skipping the check) and overwrite that row's
        # content/metadata — the tenant column itself is only ever set in
        # ON CREATE, so a race can't move a node to a different tenant, but it
        # can still let one tenant's write land on another tenant's node.
        #
        # Not fixed here: Kuzu allows only one write transaction in the whole
        # database at a time (confirmed empirically — a second connection's
        # BEGIN TRANSACTION while one is open raises immediately rather than
        # queuing), so wrapping this precheck+MERGE pair in an explicit
        # transaction would turn concurrent writers that work fine today into
        # ones that intermittently raise, which needs retry-on-conflict logic
        # on every caller to not be a regression. Given how this is actually
        # reached (two different tenants racing to create the very same node
        # id — practically only possible with a caller-supplied, not
        # library-generated, id, since consolidated fact ids now include the
        # tenant), that trade isn't worth making blind. If this needs closing,
        # do it with a retry loop around a real transaction, verified under
        # concurrent load, not as a drive-by here.
        existing = conn.execute(
            "MATCH (n:MemoryNode {id: $id}) RETURN n.tenant",
            parameters={"id": node["id"]},
        )
        assert not isinstance(existing, list)
        if existing.has_next() and existing.get_next()[0] != tenant:
            raise TenantMismatchError(
                f"node {node['id']!r} belongs to a different tenant; "
                "refusing to overwrite it"
            )
        conn.execute(
            """
            MERGE (n:MemoryNode {id: $id})
            ON CREATE SET
                n.node_type = $node_type,
                n.content = $content,
                n.embedding = $embedding,
                n.utility_score = $utility_score,
                n.access_count = $access_count,
                n.created_at = $created_at,
                n.last_accessed = $last_accessed,
                n.metadata = $metadata,
                n.valid_from = $valid_from,
                n.valid_until = $valid_until,
                n.tenant = $tenant
            ON MATCH SET
                n.node_type = $node_type,
                n.content = $content,
                n.embedding = $embedding,
                n.utility_score = $utility_score,
                n.access_count = $access_count,
                n.last_accessed = $last_accessed,
                n.metadata = $metadata,
                n.valid_from = $valid_from,
                n.valid_until = $valid_until
            """,
            parameters={
                "id": node["id"],
                "node_type": node["node_type"],
                "content": node["content"],
                "embedding": embedding_val,
                "utility_score": float(node.get("utility_score", 1.0)),
                "access_count": int(node.get("access_count", 0)),
                "created_at": node.get("created_at", _now_iso()),
                "last_accessed": node.get("last_accessed", _now_iso()),
                "metadata": json.dumps(node.get("metadata", {})),
                "valid_from": node.get("valid_from"),
                "valid_until": node.get("valid_until"),
                "tenant": tenant,
            },
        )

    def save_nodes_batch(self, nodes: list[dict[str, Any]]) -> None:
        """Upsert multiple nodes using a single connection."""
        if not nodes:
            return
        conn = self._conn()
        for node in nodes:
            self._write_node(conn, node)

    def save_node(self, node: dict[str, Any]) -> None:
        """Upsert a node using MERGE."""
        self._write_node(self._conn(), node)

    def load_node(self, node_id: str) -> dict[str, Any] | None:
        result = self._conn().execute(
            """
            MATCH (n:MemoryNode {id: $id})
            RETURN """ + _node_return() + """
            """,
            parameters={"id": node_id},
        )
        assert not isinstance(result, list)  # single statement -> one QueryResult
        if not result.has_next():
            return None
        row = result.get_next()
        return self._row_to_node(row)

    # --- Edge operations ---

    def _write_edge(self, conn: kuzu.Connection, edge: dict[str, Any]) -> None:
        """Create one edge between existing nodes on the given connection."""
        conn.execute(
            """
            MATCH (s:MemoryNode {id: $source_id}), (t:MemoryNode {id: $target_id})
            CREATE (s)-[:MemoryEdge {
                id: $id,
                relation_type: $relation_type,
                weight: $weight,
                created_at: $created_at,
                metadata: $metadata
            }]->(t)
            """,
            parameters={
                "source_id": edge["source_id"],
                "target_id": edge["target_id"],
                "id": edge["id"],
                "relation_type": edge["relation_type"],
                "weight": float(edge.get("weight", 1.0)),
                "created_at": edge.get("created_at", _now_iso()),
                "metadata": json.dumps(edge.get("metadata", {})),
            },
        )

    def save_edge(self, edge: dict[str, Any]) -> None:
        """Create an edge between existing nodes."""
        self._write_edge(self._conn(), edge)

    def write_batch(
        self, nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
    ) -> None:
        """Write nodes then edges atomically in one transaction.

        For multi-node structures (an episode = 3 nodes + edges) that must not
        be sheared by a mid-write failure. Rolls back on any error.
        """
        conn = self._conn()
        conn.execute("BEGIN TRANSACTION")
        try:
            for node in nodes:
                self._write_node(conn, node)
            for edge in edges:
                self._write_edge(conn, edge)
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

    def load_edges(
        self, node_id: str, edge_type: str | None = None
    ) -> list[dict[str, Any]]:
        conn = self._conn()
        if edge_type:
            result = conn.execute(
                """
                MATCH (s:MemoryNode)-[e:MemoryEdge]->(t:MemoryNode)
                WHERE (s.id = $id OR t.id = $id) AND e.relation_type = $etype
                RETURN e.id, s.id, t.id, e.relation_type, e.weight,
                       e.created_at, e.metadata
                """,
                parameters={"id": node_id, "etype": edge_type},
            )
        else:
            result = conn.execute(
                """
                MATCH (s:MemoryNode)-[e:MemoryEdge]->(t:MemoryNode)
                WHERE s.id = $id OR t.id = $id
                RETURN e.id, s.id, t.id, e.relation_type, e.weight,
                       e.created_at, e.metadata
                """,
                parameters={"id": node_id},
            )
        return [self._row_to_edge(r) for r in self._collect_rows(result)]

    # --- Query ---

    def query_nodes(
        self,
        node_type: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        *,
        tenant: TenantFilter = ALL_TENANTS,
    ) -> list[dict[str, Any]]:
        conn = self._conn()

        conditions: list[str] = []
        params: dict[str, Any] = {}
        if node_type:
            conditions.append("n.node_type = $ntype")
            params["ntype"] = node_type
        tenant_cond = _tenant_condition("n", tenant, params)
        if tenant_cond:
            conditions.append(tenant_cond)

        # Metadata filters are applied in Python after rows materialize. When
        # they are present the LIMIT must apply *after* filtering (parity with
        # the Postgres pushdown), so we fetch the ordered candidate set here and
        # truncate below rather than letting Cypher limit pre-filter rows.
        # The tenant filter is in the Cypher WHERE, so it always applies first.
        if filters:
            limit_clause = ""
        else:
            limit_clause = "LIMIT $lim"
            params["lim"] = limit

        result = conn.execute(
            f"""
            MATCH (n:MemoryNode)
            {_where(conditions)}
            RETURN {_node_return()}
            ORDER BY n.utility_score DESC
            {limit_clause}
            """,
            parameters=params,
        )

        nodes = [self._row_to_node(r) for r in self._collect_rows(result)]

        if filters:
            nodes = [
                n for n in nodes
                if all(
                    (n.get("metadata") or {}).get(k) == v
                    for k, v in filters.items()
                )
            ]
            nodes = nodes[:limit]

        return nodes

    def list_tenants(self) -> list[str]:
        """Every tenant that owns at least one node (untenanted excluded)."""
        result = self._conn().execute(
            "MATCH (n:MemoryNode) WHERE n.tenant IS NOT NULL "
            "RETURN DISTINCT n.tenant ORDER BY n.tenant"
        )
        return [r[0] for r in self._collect_rows(result)]

    # --- Server-side decay & evict ---

    def decay_all_scores(
        self,
        decay_rate: float,
        now_epoch_ms: int,
        *,
        tenant: TenantFilter = ALL_TENANTS,
    ) -> None:
        """Recompute utility_score for all nodes in a single Cypher SET.

        Runs entirely server-side — no node data leaves the DB.
        Formula mirrors ``synap._utils.compute_decay_score``: recency
        (``pow(1 - rate, hours)``) multiplied by ``1 + frequency_bonus`` so a
        cold node scores ~0 regardless of how often it was accessed.
        """
        params: dict[str, Any] = {"now_ms": now_epoch_ms, "rate": decay_rate}
        tenant_cond = _tenant_condition("n", tenant, params)
        self._conn().execute(
            f"""
            MATCH (n:MemoryNode)
            {_where([tenant_cond] if tenant_cond else [])}
            WITH n,
                 (cast($now_ms as DOUBLE)
                  - cast(to_epoch_ms(cast(n.last_accessed as TIMESTAMP)) as DOUBLE))
                 / 3600000.0 AS hours_raw
            WITH n,
                 CASE WHEN hours_raw < 0.000278 THEN 0.000278
                      ELSE hours_raw END AS hours
            SET n.utility_score =
                pow(1.0 - $rate, hours)
                * (1.0 + CASE WHEN cast(n.access_count AS DOUBLE) / 20.0 < 1.0
                              THEN cast(n.access_count AS DOUBLE) / 20.0
                              ELSE 1.0 END)
            """,
            parameters=params,
        )

    # --- Delete ---

    def _delete_node_on(self, conn: kuzu.Connection, node_id: str) -> None:
        """Delete a node and its edges on the given connection."""
        # Delete connected edges first (Kùzu requires directed deletes)
        conn.execute(
            "MATCH (n:MemoryNode {id: $id})-[e:MemoryEdge]->() DELETE e",
            parameters={"id": node_id},
        )
        conn.execute(
            "MATCH ()-[e:MemoryEdge]->(n:MemoryNode {id: $id}) DELETE e",
            parameters={"id": node_id},
        )
        conn.execute(
            "MATCH (n:MemoryNode {id: $id}) DELETE n",
            parameters={"id": node_id},
        )

    def delete_node(self, node_id: str) -> None:
        self._delete_node_on(self._conn(), node_id)

    def delete_nodes_batch(self, node_ids: list[str]) -> None:
        """Delete multiple nodes (and their edges) atomically in one transaction.

        Used to evict a whole episode all-or-nothing, so eviction can't shear an
        episode the way per-node deletes could.
        """
        if not node_ids:
            return
        conn = self._conn()
        conn.execute("BEGIN TRANSACTION")
        try:
            for node_id in node_ids:
                self._delete_node_on(conn, node_id)
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            raise

    def delete_edge(self, edge_id: str) -> None:
        self._conn().execute(
            """
            MATCH ()-[e:MemoryEdge {id: $id}]->()
            DELETE e
            """,
            parameters={"id": edge_id},
        )

    # --- Counts ---

    def node_count(
        self, node_type: str | None = None, *, tenant: TenantFilter = ALL_TENANTS
    ) -> int:
        conditions: list[str] = []
        params: dict[str, Any] = {}
        if node_type:
            conditions.append("n.node_type = $ntype")
            params["ntype"] = node_type
        tenant_cond = _tenant_condition("n", tenant, params)
        if tenant_cond:
            conditions.append(tenant_cond)
        result = self._conn().execute(
            f"MATCH (n:MemoryNode) {_where(conditions)} RETURN count(n)",
            parameters=params,
        )
        assert not isinstance(result, list)  # single statement -> one QueryResult
        return result.get_next()[0] if result.has_next() else 0

    def edge_count(
        self,
        relation_type: str | None = None,
        *,
        tenant: TenantFilter = ALL_TENANTS,
    ) -> int:
        # An edge belongs to its source node's tenant; edges never join two
        # tenants (the graph layer refuses that write).
        conditions: list[str] = []
        params: dict[str, Any] = {}
        if relation_type:
            conditions.append("e.relation_type = $rtype")
            params["rtype"] = relation_type
        tenant_cond = _tenant_condition("s", tenant, params)
        if tenant_cond:
            conditions.append(tenant_cond)
        result = self._conn().execute(
            f"MATCH (s:MemoryNode)-[e:MemoryEdge]->() {_where(conditions)} "
            "RETURN count(e)",
            parameters=params,
        )
        assert not isinstance(result, list)  # single statement -> one QueryResult
        return result.get_next()[0] if result.has_next() else 0

    # --- Vector similarity search ---

    def similarity_search(
        self,
        embedding: list[float],
        node_type: str | None = None,
        limit: int = 10,
        *,
        tenant: TenantFilter = ALL_TENANTS,
    ) -> list[dict[str, Any]]:
        """Native cosine similarity via Kùzu's array_cosine_similarity."""
        if len(embedding) != self._embedding_dim:
            raise ValueError(
                f"query embedding has {len(embedding)} dims but this backend is "
                f"configured for {self._embedding_dim}"
            )
        cast_expr = f"cast($emb, 'DOUBLE[{self._embedding_dim}]')"
        conditions = ["n.embedding IS NOT NULL"]
        params: dict[str, Any] = {
            "emb": [float(x) for x in embedding],
            "lim": limit,
        }
        if node_type:
            conditions.append("n.node_type = $ntype")
            params["ntype"] = node_type
        tenant_cond = _tenant_condition("n", tenant, params)
        if tenant_cond:
            conditions.append(tenant_cond)

        result = self._conn().execute(
            f"""
            MATCH (n:MemoryNode)
            {_where(conditions)}
            WITH n, array_cosine_similarity(n.embedding, {cast_expr}) AS sim
            RETURN {_node_return()}, sim
            ORDER BY sim DESC
            LIMIT $lim
            """,
            parameters=params,
        )

        return [self._row_to_node(r) for r in self._collect_rows(result)]

    # --- Graph-native traversal ---

    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_depth: int = 2,
        max_nodes: int = 50,
        min_weight: float = 0.0,
        *,
        tenant: TenantFilter = ALL_TENANTS,
    ) -> list[dict[str, Any]]:
        """BFS traversal from a start node using Cypher variable-length paths.

        Returns nodes reachable within max_depth hops, optionally
        filtered by edge relation_type and minimum weight.

        Kuzu recursive relationships bind as RECURSIVE_REL (path type),
        so edge property filters use the inline predicate syntax:
        ``[r:MemoryEdge*1..N (edge, _ | WHERE predicate)]``
        """
        # Build inline edge predicate for recursive relationship
        edge_predicates: list[str] = []
        params: dict[str, Any] = {"start": start_id, "lim": max_nodes}

        if edge_types:
            # Kuzu inline predicate can reference the single-hop edge variable
            edge_predicates.append("edge.relation_type IN $etypes")
            params["etypes"] = edge_types

        if min_weight > 0:
            edge_predicates.append(f"edge.weight >= {min_weight}")

        if edge_predicates:
            predicate_str = " AND ".join(edge_predicates)
            rel_pattern = f"[r:MemoryEdge*1..{max_depth} (edge, _ | WHERE {predicate_str})]"
        else:
            rel_pattern = f"[:MemoryEdge*1..{max_depth}]"

        # Filtering the reached nodes is enough for isolation: edges never
        # join two tenants (the graph layer refuses that write).
        neighbor_conditions = ["neighbor.id <> $start"]
        tenant_cond = _tenant_condition("neighbor", tenant, params)
        if tenant_cond:
            neighbor_conditions.append(tenant_cond)

        result = self._conn().execute(
            f"""
            MATCH (start:MemoryNode {{id: $start}})
            MATCH (start)-{rel_pattern}-(neighbor:MemoryNode)
            {_where(neighbor_conditions)}
            WITH DISTINCT neighbor
            RETURN {_node_return("neighbor")}
            LIMIT $lim
            """,
            parameters=params,
        )

        return [self._row_to_node(r) for r in self._collect_rows(result)]

    # --- Helpers ---

    def _format_embedding(self, embedding: list[float] | None) -> list[float] | None:
        if embedding is None:
            return None
        emb = [float(x) for x in embedding]
        if len(emb) != self._embedding_dim:
            raise ValueError(
                f"embedding has {len(emb)} dims but this backend is configured "
                f"for {self._embedding_dim}; refusing to store a truncated or "
                f"padded vector (was the embedder or embedding_dim misconfigured?)"
            )
        return emb

    def _row_to_node(self, row: list) -> dict[str, Any]:
        return {
            "id": row[0],
            "node_type": row[1],
            "content": row[2],
            "embedding": list(row[3]) if row[3] is not None else None,
            "utility_score": row[4],
            "access_count": row[5],
            "created_at": row[6],
            "last_accessed": row[7],
            "metadata": json.loads(row[8]) if isinstance(row[8], str) else row[8],
            "valid_from": row[9] if len(row) > 9 else None,
            "valid_until": row[10] if len(row) > 10 else None,
            "tenant": row[11] if len(row) > 11 else None,
        }

    def _row_to_edge(self, row: list) -> dict[str, Any]:
        return {
            "id": row[0],
            "source_id": row[1],
            "target_id": row[2],
            "relation_type": row[3],
            "weight": row[4],
            "created_at": row[5],
            "metadata": json.loads(row[6]) if isinstance(row[6], str) else row[6],
        }

    def _collect_rows(self, result) -> list[list]:
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return rows

    def close(self) -> None:
        # Kùzu handles cleanup on garbage collection
        self._db = None
