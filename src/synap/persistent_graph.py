"""PersistentGraph — async storage-backed graph for sync and async backends."""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timezone
from typing import Any

from synap._utils import compute_decay_score, select_evictions
from synap.protocols import AsyncStorageBackend, StorageBackend
from synap.types import MemoryEdge, MemoryNode, MemoryType

# Re-exported for callers that import it from here; canonical home is synap._utils.
__all__ = ["PersistentGraph", "compute_decay_score"]


def _node_to_dict(node: MemoryNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "node_type": node.node_type.value,
        "content": node.content,
        "embedding": node.embedding,
        "utility_score": node.utility_score,
        "access_count": node.access_count,
        "created_at": node.created_at.isoformat(),
        "last_accessed": node.last_accessed.isoformat(),
        "metadata": node.metadata,
        "valid_from": node.valid_from.isoformat() if node.valid_from else None,
        "valid_until": node.valid_until.isoformat() if node.valid_until else None,
    }


def _dict_to_node(d: dict[str, Any]) -> MemoryNode:
    return MemoryNode(
        id=d["id"],
        node_type=MemoryType(d["node_type"]),
        content=d["content"],
        embedding=d.get("embedding"),
        utility_score=d.get("utility_score", 1.0),
        access_count=d.get("access_count", 0),
        created_at=_parse_dt(d.get("created_at")),
        last_accessed=_parse_dt(d.get("last_accessed")),
        metadata=d.get("metadata") if isinstance(d.get("metadata"), dict) else {},
        valid_from=_parse_dt_optional(d.get("valid_from")),
        valid_until=_parse_dt_optional(d.get("valid_until")),
    )


def _edge_to_dict(edge: MemoryEdge) -> dict[str, Any]:
    return {
        "id": edge.id,
        "source_id": edge.source_id,
        "target_id": edge.target_id,
        "relation_type": edge.relation_type,
        "weight": edge.weight,
        "created_at": edge.created_at.isoformat(),
        "metadata": edge.metadata,
    }


def _dict_to_edge(d: dict[str, Any]) -> MemoryEdge:
    return MemoryEdge(
        id=d["id"],
        source_id=d["source_id"],
        target_id=d["target_id"],
        relation_type=d["relation_type"],
        weight=d.get("weight", 1.0),
        created_at=_parse_dt(d.get("created_at")),
        metadata=d.get("metadata") if isinstance(d.get("metadata"), dict) else {},
    )


def _parse_dt(val: Any) -> datetime:
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        return datetime.fromisoformat(val)
    return datetime.now(timezone.utc)


def _parse_dt_optional(val: Any) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        return datetime.fromisoformat(val)
    return None


class PersistentGraph:
    """Async storage-backed graph for both sync and async backends.

    Sync backends (Kuzu, SQLite) are dispatched via asyncio.to_thread.
    Async backends (Postgres) are called directly.
    """

    def __init__(
        self,
        backend: StorageBackend | AsyncStorageBackend,
        utility_decay_rate: float = 0.01,
    ) -> None:
        self._backend = backend
        self._utility_decay_rate = utility_decay_rate
        self._async = inspect.iscoroutinefunction(
            getattr(backend, "save_node", None)
        )

    async def _call(self, method: Any, *args: Any, **kwargs: Any) -> Any:
        """Dispatch to backend: await if async, to_thread if sync."""
        if self._async:
            return await method(*args, **kwargs)
        return await asyncio.to_thread(method, *args, **kwargs)

    @property
    def backend(self) -> StorageBackend | AsyncStorageBackend:
        return self._backend

    # --- Node operations ---

    async def add_node(self, node: MemoryNode) -> str:
        await self._call(self._backend.save_node, _node_to_dict(node))
        return node.id

    async def write_batch(
        self, nodes: list[MemoryNode], edges: list[MemoryEdge]
    ) -> None:
        """Persist nodes then edges in one transaction when the backend supports it.

        Falls back to sequential (non-atomic) writes for backends without
        write_batch — the in-memory graph and any future minimal backend.
        """
        node_dicts = [_node_to_dict(n) for n in nodes]
        edge_dicts = [_edge_to_dict(e) for e in edges]
        if hasattr(self._backend, "write_batch"):
            await self._call(self._backend.write_batch, node_dicts, edge_dicts)
        else:
            for d in node_dicts:
                await self._call(self._backend.save_node, d)
            for d in edge_dicts:
                await self._call(self._backend.save_edge, d)

    async def get_node(self, node_id: str) -> MemoryNode | None:
        d = await self._call(self._backend.load_node, node_id)
        if d is None:
            return None
        return _dict_to_node(d)

    async def remove_node(self, node_id: str) -> None:
        await self._call(self._backend.delete_node, node_id)

    async def node_count(self, node_type: MemoryType | None = None) -> int:
        return await self._call(
            self._backend.node_count,
            node_type.value if node_type else None,
        )

    # --- Edge operations ---

    async def add_edge(self, edge: MemoryEdge) -> str:
        src = await self._call(self._backend.load_node, edge.source_id)
        if src is None:
            raise KeyError(f"Source node {edge.source_id} not in graph")
        tgt = await self._call(self._backend.load_node, edge.target_id)
        if tgt is None:
            raise KeyError(f"Target node {edge.target_id} not in graph")
        await self._call(self._backend.save_edge, _edge_to_dict(edge))
        return edge.id

    async def remove_edge(self, edge_id: str) -> None:
        await self._call(self._backend.delete_edge, edge_id)

    async def edge_count(self, relation_type: str | None = None) -> int:
        return await self._call(self._backend.edge_count, relation_type)

    # --- Traversal ---

    async def neighbors(
        self,
        node_id: str,
        edge_type: str | None = None,
        direction: str = "outgoing",
    ) -> list[MemoryNode]:
        edges = await self._call(self._backend.load_edges, node_id, edge_type)
        results = []
        for e in edges:
            if direction == "outgoing" and e["source_id"] != node_id:
                continue
            if direction == "incoming" and e["target_id"] != node_id:
                continue
            neighbor_id = (
                e["target_id"] if e["source_id"] == node_id else e["source_id"]
            )
            d = await self._call(self._backend.load_node, neighbor_id)
            if d:
                results.append(_dict_to_node(d))
        return results

    async def traverse(
        self,
        start: str,
        edge_types: list[str] | None = None,
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> list[MemoryNode]:
        results = await self._call(
            self._backend.traverse, start, edge_types, max_depth, max_nodes,
        )
        return [_dict_to_node(d) for d in results]

    # --- Query ---

    async def query(
        self,
        node_type: MemoryType | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[MemoryNode]:
        results = await self._call(
            self._backend.query_nodes,
            node_type.value if node_type else None,
            filters,
            limit,
        )
        return [_dict_to_node(d) for d in results]

    # --- Utility & lifecycle ---

    async def update_utility(self, node_id: str) -> None:
        d = await self._call(self._backend.load_node, node_id)
        if d is None:
            return
        node = _dict_to_node(d)
        node.touch()  # sets last_accessed = now
        seconds = max(
            1.0,
            (datetime.now(timezone.utc) - node.last_accessed).total_seconds(),
        )
        node.utility_score = compute_decay_score(
            seconds / 3600, node.access_count, self._utility_decay_rate
        )
        await self._call(self._backend.save_node, _node_to_dict(node))

    async def decay_all(self) -> None:
        # Server-side path: single Cypher SET, no data leaves the DB
        if hasattr(self._backend, "decay_all_scores"):
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            await self._call(
                self._backend.decay_all_scores,
                self._utility_decay_rate,
                now_ms,
            )
            return

        # Fallback for backends without server-side decay
        now = datetime.now(timezone.utc)
        all_nodes = await self._call(
            self._backend.query_nodes, None, None, 100_000
        )
        updated = []
        for d in all_nodes:
            node = _dict_to_node(d)
            seconds = max(1.0, (now - node.last_accessed).total_seconds())
            node.utility_score = compute_decay_score(
                seconds / 3600, node.access_count, self._utility_decay_rate
            )
            updated.append(_node_to_dict(node))

        if hasattr(self._backend, "save_nodes_batch"):
            await self._call(self._backend.save_nodes_batch, updated)
        else:
            for d in updated:
                await self._call(self._backend.save_node, d)

    async def evict(self, threshold: float = 0.1) -> list[str]:
        # Episode-aware eviction: whole episode or nothing (see select_evictions).
        # This can't stay server-side — Kuzu stores metadata as a JSON string, so
        # grouping episodic nodes by episode_id happens here, not in the query.
        all_nodes = await self._call(
            self._backend.query_nodes, None, None, 100_000
        )
        items = [
            (
                d["id"],
                d.get("utility_score", 1.0),
                (d.get("metadata") or {}).get("episode_id"),
            )
            for d in all_nodes
        ]
        to_evict = select_evictions(items, threshold)
        if to_evict:
            if hasattr(self._backend, "delete_nodes_batch"):
                await self._call(self._backend.delete_nodes_batch, to_evict)
            else:
                for nid in to_evict:
                    await self._call(self._backend.delete_node, nid)
        return to_evict

    # --- Edges between specific nodes ---

    async def edges_between(
        self,
        source_id: str,
        target_id: str,
        relation_type: str | None = None,
    ) -> list[MemoryEdge]:
        edges = await self._call(self._backend.load_edges, source_id)
        results = []
        for e in edges:
            if e["source_id"] != source_id or e["target_id"] != target_id:
                continue
            if relation_type and e["relation_type"] != relation_type:
                continue
            results.append(_dict_to_edge(e))
        return results

    async def has_incoming_edge(self, node_id: str, relation_type: str) -> bool:
        edges = await self._call(self._backend.load_edges, node_id)
        for e in edges:
            if e["target_id"] == node_id and e["relation_type"] == relation_type:
                return True
        return False

    async def similarity_search(
        self,
        embedding: list[float],
        node_type: MemoryType | None = None,
        limit: int = 10,
    ) -> list[MemoryNode]:
        results = await self._call(
            self._backend.similarity_search,
            embedding,
            node_type.value if node_type else None,
            limit,
        )
        return [_dict_to_node(d) for d in results]

    def close(self) -> None:
        if hasattr(self._backend, "close"):
            self._backend.close()
