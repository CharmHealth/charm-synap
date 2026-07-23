"""Provider protocols — interfaces the consumer implements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from synap.types import DomainResult, MemoryEdge, MemoryNode, MemoryType


class EmbeddingProvider(Protocol):
    """Embeds text into vector space for entry-point matching."""

    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class LLMProvider(Protocol):
    """Generates text for consolidation and bootstrapping."""

    async def generate(
        self,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
    ) -> str: ...


class StorageBackend(Protocol):
    """Persistent storage for the memory graph.

    Stays synchronous — backends are local/embedded (no network I/O).
    PersistentGraph wraps calls with asyncio.to_thread.
    """

    def save_node(self, node: dict[str, Any]) -> None: ...
    def save_edge(self, edge: dict[str, Any]) -> None: ...
    def load_node(self, node_id: str) -> dict[str, Any] | None: ...
    def load_edges(
        self, node_id: str, edge_type: str | None = None
    ) -> list[dict[str, Any]]: ...
    def query_nodes(
        self,
        node_type: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]: ...
    def delete_node(self, node_id: str) -> None: ...
    def delete_edge(self, edge_id: str) -> None: ...
    def node_count(self, node_type: str | None = None) -> int: ...
    def edge_count(self, relation_type: str | None = None) -> int: ...
    def similarity_search(
        self,
        embedding: list[float],
        node_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]: ...
    def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> list[dict[str, Any]]: ...


class AsyncStorageBackend(Protocol):
    """Async persistent storage for the memory graph.

    For network-backed databases (Postgres, Neo4j) where async I/O matters.
    PersistentGraph calls methods directly without asyncio.to_thread.
    """

    async def save_node(self, node: dict[str, Any]) -> None: ...
    async def save_edge(self, edge: dict[str, Any]) -> None: ...
    async def load_node(self, node_id: str) -> dict[str, Any] | None: ...
    async def load_edges(
        self, node_id: str, edge_type: str | None = None
    ) -> list[dict[str, Any]]: ...
    async def query_nodes(
        self,
        node_type: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]: ...
    async def delete_node(self, node_id: str) -> None: ...
    async def delete_edge(self, edge_id: str) -> None: ...
    async def node_count(self, node_type: str | None = None) -> int: ...
    async def edge_count(self, relation_type: str | None = None) -> int: ...
    async def similarity_search(
        self,
        embedding: list[float],
        node_type: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]: ...
    async def traverse(
        self,
        start_id: str,
        edge_types: list[str] | None = None,
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> list[dict[str, Any]]: ...


class SemanticDomain(Protocol):
    """Domain-specific semantic knowledge adapter.

    The consuming project implements this to define how domain knowledge
    is stored and retrieved. synap handles lifecycle (when to retrieve,
    when to consolidate); the domain handles shape (what types exist,
    how to query them).
    """

    async def retrieve(
        self,
        task_description: str,
        task_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        retrieval_hints: dict[str, Any] | None = None,
    ) -> list[DomainResult]: ...

    async def absorb(
        self,
        insights: list[str],
        source_episodes: list[MemoryNode],
        metadata: dict[str, Any] | None = None,
    ) -> str | None: ...


class GraphStore(Protocol):
    """Async interface for the graph layer used by subsystems.

    Both MemoryGraph (in-memory) and PersistentGraph (backend-backed)
    implement this protocol.
    """

    async def add_node(self, node: MemoryNode) -> str: ...
    async def get_node(self, node_id: str) -> MemoryNode | None: ...
    async def remove_node(self, node_id: str) -> None: ...
    async def node_count(self, node_type: MemoryType | None = None) -> int: ...
    async def add_edge(self, edge: MemoryEdge) -> str: ...
    async def write_batch(
        self, nodes: list[MemoryNode], edges: list[MemoryEdge]
    ) -> None: ...
    async def remove_edge(self, edge_id: str) -> None: ...
    async def edge_count(self, relation_type: str | None = None) -> int: ...
    async def traverse(
        self,
        start: str,
        edge_types: list[str] | None = None,
        max_depth: int = 2,
        max_nodes: int = 50,
    ) -> list[MemoryNode]: ...
    async def query(
        self,
        node_type: MemoryType | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> list[MemoryNode]: ...
    async def update_utility(self, node_id: str) -> None: ...
    async def decay_all(self) -> None: ...
    async def evict(self, threshold: float = 0.1) -> list[str]: ...
    async def edges_between(
        self,
        source_id: str,
        target_id: str,
        relation_type: str | None = None,
    ) -> list[MemoryEdge]: ...
    async def has_incoming_edge(self, node_id: str, relation_type: str) -> bool: ...
    async def similarity_search(
        self,
        embedding: list[float],
        node_type: MemoryType | None = None,
        limit: int = 10,
    ) -> list[MemoryNode]: ...
