"""SQLite storage backend for persistent memory graphs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from synap._utils import cosine_similarity


class SQLiteBackend:
    """Persistent storage using SQLite.

    Stores nodes and edges as JSON blobs with indexed columns
    for common query patterns. Embedding similarity search uses
    brute-force cosine similarity (adequate for <100K nodes;
    swap to a vector extension for larger stores).
    """

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._conn = sqlite3.connect(self._path)
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
        self._conn.commit()

    def save_node(self, node: dict[str, Any]) -> None:
        embedding_json = (
            json.dumps(node.get("embedding"))
            if node.get("embedding") is not None
            else None
        )
        self._conn.execute(
            """INSERT OR REPLACE INTO nodes
               (id, node_type, content, embedding, utility_score,
                access_count, created_at, last_accessed, metadata, data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
            ),
        )
        self._conn.commit()

    def save_edge(self, edge: dict[str, Any]) -> None:
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
        self._conn.commit()

    def load_node(self, node_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT data FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["data"])

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

    def query_nodes(
        self,
        node_type: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        query = "SELECT data, metadata FROM nodes WHERE 1=1"
        params: list[Any] = []

        if node_type:
            query += " AND node_type = ?"
            params.append(node_type)

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
            data = json.loads(row["data"])
            if filters:
                meta = json.loads(row["metadata"])
                if all(meta.get(k) == v for k, v in filters.items()):
                    results.append(data)
                    if len(results) >= limit:
                        break
            else:
                results.append(data)
        return results

    def node_count(self, node_type: str | None = None) -> int:
        if node_type:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM nodes WHERE node_type = ?", (node_type,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()
        return row[0]

    def edge_count(self, relation_type: str | None = None) -> int:
        if relation_type:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM edges WHERE relation_type = ?",
                (relation_type,),
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()
        return row[0]

    def delete_node(self, node_id: str) -> None:
        self._conn.execute(
            "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
            (node_id, node_id),
        )
        self._conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        self._conn.commit()

    def delete_edge(self, edge_id: str) -> None:
        self._conn.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
        self._conn.commit()

    def similarity_search(
        self,
        embedding: list[float],
        node_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Brute-force cosine similarity search."""
        query = "SELECT data, embedding FROM nodes WHERE embedding IS NOT NULL"
        params: list[Any] = []
        if node_type:
            query += " AND node_type = ?"
            params.append(node_type)

        rows = self._conn.execute(query, params).fetchall()
        scored: list[tuple[float, dict[str, Any]]] = []

        for row in rows:
            node_embedding = json.loads(row["embedding"])
            sim = cosine_similarity(embedding, node_embedding)
            scored.append((sim, json.loads(row["data"])))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [data for _, data in scored[:limit]]

    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> list[dict[str, Any]]:
        """BFS traversal from start node. Not graph-native — walks edges in Python."""
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
                        if node:
                            results.append(node)
                            if len(results) >= max_nodes:
                                return results
                            next_frontier.append(neighbor_id)
            frontier = next_frontier
            if not frontier:
                break

        return results

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
