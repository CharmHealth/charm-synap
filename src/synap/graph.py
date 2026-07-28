"""MemoryGraph — async in-memory typed property graph backing all three memory subsystems."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from synap._utils import compute_decay_score, cosine_similarity, select_evictions
from synap.types import MemoryEdge, MemoryNode, MemoryType


class MemoryGraph:
    """In-memory typed property graph.

    Each subsystem owns a partition of nodes (by node_type) but edges
    can cross partitions — this is how consolidation creates links
    between episodic experiences and semantic facts.

    All methods are async to satisfy the GraphStore protocol.
    In-memory operations return immediately.
    """

    def __init__(self, utility_decay_rate: float = 0.01) -> None:
        self._nodes: dict[str, MemoryNode] = {}
        self._edges: dict[str, MemoryEdge] = {}
        self._outgoing: dict[str, list[str]] = defaultdict(list)
        self._incoming: dict[str, list[str]] = defaultdict(list)
        self._utility_decay_rate = utility_decay_rate

    # --- Node operations ---

    async def add_node(self, node: MemoryNode) -> str:
        self._nodes[node.id] = node
        return node.id

    async def write_batch(
        self, nodes: list[MemoryNode], edges: list[MemoryEdge]
    ) -> None:
        """Add nodes then edges atomically: on any failure, nothing is applied."""
        node_backup = dict(self._nodes)
        edge_backup = dict(self._edges)
        out_backup = defaultdict(list, {k: list(v) for k, v in self._outgoing.items()})
        in_backup = defaultdict(list, {k: list(v) for k, v in self._incoming.items()})
        try:
            for node in nodes:
                await self.add_node(node)
            for edge in edges:
                await self.add_edge(edge)
        except Exception:
            self._nodes = node_backup
            self._edges = edge_backup
            self._outgoing = out_backup
            self._incoming = in_backup
            raise

    async def get_node(self, node_id: str) -> MemoryNode | None:
        return self._nodes.get(node_id)

    async def remove_node(self, node_id: str) -> None:
        for edge_id in list(self._outgoing.get(node_id, [])):
            await self.remove_edge(edge_id)
        for edge_id in list(self._incoming.get(node_id, [])):
            await self.remove_edge(edge_id)
        self._outgoing.pop(node_id, None)
        self._incoming.pop(node_id, None)
        self._nodes.pop(node_id, None)

    async def node_count(self, node_type: MemoryType | None = None) -> int:
        if node_type is None:
            return len(self._nodes)
        return sum(1 for n in self._nodes.values() if n.node_type == node_type)

    # --- Edge operations ---

    async def add_edge(self, edge: MemoryEdge) -> str:
        if edge.source_id not in self._nodes:
            raise KeyError(f"Source node {edge.source_id} not in graph")
        if edge.target_id not in self._nodes:
            raise KeyError(f"Target node {edge.target_id} not in graph")
        self._edges[edge.id] = edge
        self._outgoing[edge.source_id].append(edge.id)
        self._incoming[edge.target_id].append(edge.id)
        return edge.id

    async def get_edge(self, edge_id: str) -> MemoryEdge | None:
        return self._edges.get(edge_id)

    async def remove_edge(self, edge_id: str) -> None:
        edge = self._edges.pop(edge_id, None)
        if edge is None:
            return
        out = self._outgoing.get(edge.source_id, [])
        if edge_id in out:
            out.remove(edge_id)
        inc = self._incoming.get(edge.target_id, [])
        if edge_id in inc:
            inc.remove(edge_id)

    async def edge_count(self, relation_type: str | None = None) -> int:
        if relation_type is None:
            return len(self._edges)
        return sum(1 for e in self._edges.values() if e.relation_type == relation_type)

    # --- Traversal ---

    async def neighbors(
        self,
        node_id: str,
        edge_type: str | None = None,
        direction: str = "outgoing",
    ) -> list[MemoryNode]:
        if direction == "outgoing":
            edge_ids = self._outgoing.get(node_id, [])
            get_neighbor_id = lambda e: e.target_id
        elif direction == "incoming":
            edge_ids = self._incoming.get(node_id, [])
            get_neighbor_id = lambda e: e.source_id
        else:
            edge_ids = self._outgoing.get(node_id, []) + self._incoming.get(
                node_id, []
            )
            get_neighbor_id = (
                lambda e: e.target_id if e.source_id == node_id else e.source_id
            )

        results = []
        for eid in edge_ids:
            edge = self._edges.get(eid)
            if edge is None:
                continue
            if edge_type is not None and edge.relation_type != edge_type:
                continue
            neighbor = self._nodes.get(get_neighbor_id(edge))
            if neighbor is not None:
                results.append(neighbor)
        return results

    async def traverse(
        self,
        start: str,
        edge_types: list[str] | None = None,
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> list[MemoryNode]:
        if start not in self._nodes:
            return []

        visited: set[str] = {start}
        results: list[MemoryNode] = []
        frontier: list[str] = [start]

        for _depth in range(max_depth):
            next_frontier: list[str] = []
            for node_id in frontier:
                for edge_id in self._outgoing.get(node_id, []) + self._incoming.get(
                    node_id, []
                ):
                    edge = self._edges.get(edge_id)
                    if edge is None:
                        continue
                    if edge_types and edge.relation_type not in edge_types:
                        continue
                    neighbor_id = (
                        edge.target_id
                        if edge.source_id == node_id
                        else edge.source_id
                    )
                    if neighbor_id in visited:
                        continue
                    visited.add(neighbor_id)
                    neighbor = self._nodes.get(neighbor_id)
                    if neighbor is not None:
                        results.append(neighbor)
                        next_frontier.append(neighbor_id)
                    if len(results) >= max_nodes:
                        return results
            frontier = next_frontier
            if not frontier:
                break

        return results

    # --- Query ---

    async def query(
        self,
        node_type: MemoryType | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[MemoryNode]:
        results: list[MemoryNode] = []
        for node in self._nodes.values():
            if node_type is not None and node.node_type != node_type:
                continue
            if filters:
                if not all(node.metadata.get(k) == v for k, v in filters.items()):
                    continue
            results.append(node)
            if len(results) >= limit:
                break
        return results

    # --- Utility & lifecycle ---

    async def update_utility(self, node_id: str) -> None:
        node = self._nodes.get(node_id)
        if node is None:
            return
        node.touch()  # sets last_accessed = now
        seconds = max(
            1.0,
            (datetime.now(timezone.utc) - node.last_accessed).total_seconds(),
        )
        node.utility_score = compute_decay_score(
            seconds / 3600, node.access_count, self._utility_decay_rate
        )

    async def decay_all(self) -> None:
        now = datetime.now(timezone.utc)
        for node in self._nodes.values():
            seconds = max(1.0, (now - node.last_accessed).total_seconds())
            node.utility_score = compute_decay_score(
                seconds / 3600, node.access_count, self._utility_decay_rate
            )

    async def evict(self, threshold: float = 0.1) -> list[str]:
        items = [
            (nid, node.utility_score, (node.metadata or {}).get("episode_id"))
            for nid, node in self._nodes.items()
        ]
        to_evict = select_evictions(items, threshold)
        for nid in to_evict:
            await self.remove_node(nid)
        return to_evict

    # --- Edges between specific nodes ---

    async def edges_between(
        self, source_id: str, target_id: str, relation_type: str | None = None
    ) -> list[MemoryEdge]:
        results = []
        for eid in self._outgoing.get(source_id, []):
            edge = self._edges.get(eid)
            if edge is None:
                continue
            if edge.target_id != target_id:
                continue
            if relation_type and edge.relation_type != relation_type:
                continue
            results.append(edge)
        return results

    async def has_incoming_edge(self, node_id: str, relation_type: str) -> bool:
        for eid in self._incoming.get(node_id, []):
            edge = self._edges.get(eid)
            if edge and edge.relation_type == relation_type:
                return True
        return False

    async def similarity_search(
        self,
        embedding: list[float],
        node_type: MemoryType | None = None,
        limit: int = 10,
    ) -> list[MemoryNode]:
        candidates: list[tuple[float, MemoryNode]] = []
        for node in self._nodes.values():
            if node_type is not None and node.node_type != node_type:
                continue
            if node.embedding is None:
                continue
            if len(node.embedding) != len(embedding):
                raise ValueError(
                    f"embedding dimension mismatch in similarity search: query has "
                    f"{len(embedding)}, node {node.id} has {len(node.embedding)}; "
                    f"the store holds inconsistent embedding dimensions"
                )
            sim = cosine_similarity(embedding, node.embedding)
            candidates.append((sim, node))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in candidates[:limit]]
