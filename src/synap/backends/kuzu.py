"""Kùzu graph database backend — native graph traversal + vector search."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import kuzu


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
    return datetime.now(timezone.utc).isoformat()


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
        self._db = kuzu.Database(self._path, buffer_pool_size=buffer_pool_mb * 1024 * 1024)
        self._ensure_schema()

    def _conn(self) -> kuzu.Connection:
        """Create a fresh connection for the current operation."""
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

    def save_nodes_batch(self, nodes: list[dict[str, Any]]) -> None:
        """Upsert multiple nodes using a single connection."""
        if not nodes:
            return
        conn = self._conn()
        for node in nodes:
            embedding = node.get("embedding")
            embedding_val = self._format_embedding(embedding) if embedding else None
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
                    n.valid_until = $valid_until
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
                },
            )

    def save_node(self, node: dict[str, Any]) -> None:
        """Upsert a node using MERGE."""
        embedding = node.get("embedding")
        embedding_val = self._format_embedding(embedding) if embedding else None

        self._conn().execute(
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
                n.valid_until = $valid_until
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
            },
        )

    def load_node(self, node_id: str) -> dict[str, Any] | None:
        result = self._conn().execute(
            """
            MATCH (n:MemoryNode {id: $id})
            RETURN n.id, n.node_type, n.content, n.embedding,
                   n.utility_score, n.access_count,
                   n.created_at, n.last_accessed, n.metadata,
                   n.valid_from, n.valid_until
            """,
            parameters={"id": node_id},
        )
        if not result.has_next():
            return None
        row = result.get_next()
        return self._row_to_node(row)

    # --- Edge operations ---

    def save_edge(self, edge: dict[str, Any]) -> None:
        """Create an edge between existing nodes."""
        self._conn().execute(
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
    ) -> list[dict[str, Any]]:
        conn = self._conn()

        where = ""
        params: dict[str, Any] = {}
        if node_type:
            where = "WHERE n.node_type = $ntype"
            params["ntype"] = node_type

        # Metadata filters are applied in Python after rows materialize. When
        # they are present the LIMIT must apply *after* filtering (parity with
        # the Postgres pushdown), so we fetch the ordered candidate set here and
        # truncate below rather than letting Cypher limit pre-filter rows.
        if filters:
            limit_clause = ""
        else:
            limit_clause = "LIMIT $lim"
            params["lim"] = limit

        result = conn.execute(
            f"""
            MATCH (n:MemoryNode)
            {where}
            RETURN n.id, n.node_type, n.content, n.embedding,
                   n.utility_score, n.access_count,
                   n.created_at, n.last_accessed, n.metadata,
                   n.valid_from, n.valid_until
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

    # --- Server-side decay & evict ---

    def decay_all_scores(self, decay_rate: float, now_epoch_ms: int) -> None:
        """Recompute utility_score for all nodes in a single Cypher SET.

        Runs entirely server-side — no node data leaves the DB.
        Formula mirrors ``persistent_graph.compute_decay_score``.
        """
        self._conn().execute(
            """
            MATCH (n:MemoryNode)
            WITH n,
                 (cast($now_ms as DOUBLE)
                  - cast(to_epoch_ms(cast(n.last_accessed as TIMESTAMP)) as DOUBLE))
                 / 3600000.0 AS hours_raw
            WITH n,
                 CASE WHEN hours_raw < 0.000278 THEN 0.000278
                      ELSE hours_raw END AS hours
            SET n.utility_score =
                pow(1.0 - $rate, hours)
                + CASE WHEN cast(n.access_count AS DOUBLE) / 20.0 < 1.0
                       THEN cast(n.access_count AS DOUBLE) / 20.0
                       ELSE 1.0 END
            """,
            parameters={
                "now_ms": now_epoch_ms,
                "rate": decay_rate,
            },
        )

    def evict_by_score(self, threshold: float) -> list[str]:
        """Delete nodes with utility_score below threshold, server-side.

        Returns IDs of evicted nodes.
        """
        conn = self._conn()
        # Collect IDs first (lightweight — no embeddings)
        result = conn.execute(
            """
            MATCH (n:MemoryNode)
            WHERE n.utility_score < $threshold
            RETURN n.id
            """,
            parameters={"threshold": threshold},
        )
        ids = [row[0] for row in self._collect_rows(result)]
        # Delete each (handles edge cleanup)
        for nid in ids:
            self.delete_node(nid)
        return ids

    # --- Delete ---

    def delete_node(self, node_id: str) -> None:
        conn = self._conn()
        # Delete connected edges first (Kùzu requires directed deletes)
        conn.execute(
            """
            MATCH (n:MemoryNode {id: $id})-[e:MemoryEdge]->()
            DELETE e
            """,
            parameters={"id": node_id},
        )
        conn.execute(
            """
            MATCH ()-[e:MemoryEdge]->(n:MemoryNode {id: $id})
            DELETE e
            """,
            parameters={"id": node_id},
        )
        conn.execute(
            """
            MATCH (n:MemoryNode {id: $id})
            DELETE n
            """,
            parameters={"id": node_id},
        )

    def delete_edge(self, edge_id: str) -> None:
        self._conn().execute(
            """
            MATCH ()-[e:MemoryEdge {id: $id}]->()
            DELETE e
            """,
            parameters={"id": edge_id},
        )

    # --- Counts ---

    def node_count(self, node_type: str | None = None) -> int:
        conn = self._conn()
        if node_type:
            result = conn.execute(
                "MATCH (n:MemoryNode) WHERE n.node_type = $ntype RETURN count(n)",
                parameters={"ntype": node_type},
            )
        else:
            result = conn.execute(
                "MATCH (n:MemoryNode) RETURN count(n)"
            )
        return result.get_next()[0] if result.has_next() else 0

    def edge_count(self, relation_type: str | None = None) -> int:
        conn = self._conn()
        if relation_type:
            result = conn.execute(
                "MATCH ()-[e:MemoryEdge]->() WHERE e.relation_type = $rtype RETURN count(e)",
                parameters={"rtype": relation_type},
            )
        else:
            result = conn.execute(
                "MATCH ()-[e:MemoryEdge]->() RETURN count(e)"
            )
        return result.get_next()[0] if result.has_next() else 0

    # --- Vector similarity search ---

    def similarity_search(
        self,
        embedding: list[float],
        node_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Native cosine similarity via Kùzu's array_cosine_similarity."""
        if len(embedding) != self._embedding_dim:
            raise ValueError(
                f"query embedding has {len(embedding)} dims but this backend is "
                f"configured for {self._embedding_dim}"
            )
        cast_expr = f"cast($emb, 'DOUBLE[{self._embedding_dim}]')"
        conn = self._conn()

        if node_type:
            result = conn.execute(
                f"""
                MATCH (n:MemoryNode)
                WHERE n.node_type = $ntype AND n.embedding IS NOT NULL
                WITH n, array_cosine_similarity(n.embedding, {cast_expr}) AS sim
                RETURN n.id, n.node_type, n.content, n.embedding,
                       n.utility_score, n.access_count,
                       n.created_at, n.last_accessed, n.metadata,
                       n.valid_from, n.valid_until, sim
                ORDER BY sim DESC
                LIMIT $lim
                """,
                parameters={
                    "ntype": node_type,
                    "emb": [float(x) for x in embedding],
                    "lim": limit,
                },
            )
        else:
            result = conn.execute(
                f"""
                MATCH (n:MemoryNode)
                WHERE n.embedding IS NOT NULL
                WITH n, array_cosine_similarity(n.embedding, {cast_expr}) AS sim
                RETURN n.id, n.node_type, n.content, n.embedding,
                       n.utility_score, n.access_count,
                       n.created_at, n.last_accessed, n.metadata,
                       n.valid_from, n.valid_until, sim
                ORDER BY sim DESC
                LIMIT $lim
                """,
                parameters={
                    "emb": [float(x) for x in embedding],
                    "lim": limit,
                },
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

        result = self._conn().execute(
            f"""
            MATCH (start:MemoryNode {{id: $start}})
            MATCH (start)-{rel_pattern}-(neighbor:MemoryNode)
            WHERE neighbor.id <> $start
            WITH DISTINCT neighbor
            RETURN neighbor.id, neighbor.node_type, neighbor.content, neighbor.embedding,
                   neighbor.utility_score, neighbor.access_count,
                   neighbor.created_at, neighbor.last_accessed, neighbor.metadata,
                   neighbor.valid_from, neighbor.valid_until
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
