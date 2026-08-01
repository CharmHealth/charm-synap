"""Shared request/response models for HTTP contrib modules.

Framework-agnostic Pydantic models used by both FastAPI and Sanic
adapters. These define the REST API contract for synap.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ── Knowledge Unit (the cross-app shared format) ──


class KnowledgeSource(BaseModel):
    """Where a knowledge unit came from."""

    app: str = Field(description="Source application (stacks, synthesis, anvil, etc.)")
    document: str = Field(description="Document title or identifier")
    author: str | None = Field(default=None, description="Author of the source material")
    location: str | None = Field(default=None, description="Page, CFI, section — format varies")
    document_type: str | None = Field(
        default=None, description="pdf, epub, handwritten, article, metric"
    )


class StoreKnowledgeRequest(BaseModel):
    """Absorb a knowledge unit into the graph."""

    content: str = Field(description="The captured text")
    domain: str = Field(description="Semantic domain: reading, current_affairs, practice, etc.")
    source: KnowledgeSource
    metadata: dict[str, Any] = Field(default_factory=dict, description="Domain-specific metadata")
    connections: list[str] = Field(
        default_factory=list, description="IDs of related knowledge units to link"
    )


class KnowledgeUnitResponse(BaseModel):
    """A stored knowledge unit."""

    id: str
    content: str
    domain: str
    source: KnowledgeSource
    metadata: dict[str, Any]
    created_at: datetime
    connections: list[str] = Field(default_factory=list)


class StoreKnowledgeResponse(BaseModel):
    """Response after storing a knowledge unit."""

    id: str
    connections_created: int = 0


# ── Search ──


class SearchRequest(BaseModel):
    """Search for knowledge units."""

    query: str
    domain: str | None = Field(default=None, description="Scope search to a specific domain")
    max_results: int = Field(default=10, ge=1, le=100)
    max_depth: int = Field(default=2, ge=1, le=5)


class SearchResultItem(BaseModel):
    """A single search result."""

    id: str
    content: str
    relevance: float
    metadata: dict[str, Any]
    domain: str | None = None
    source: KnowledgeSource | None = None


class SearchResponse(BaseModel):
    """Search results."""

    results: list[SearchResultItem]
    query: str


# ── Connect ──


class ConnectRequest(BaseModel):
    """Link two knowledge units."""

    target_id: str
    relation_type: str = Field(default="related_to")
    weight: float = Field(default=1.0, ge=0.0, le=10.0)


class ConnectResponse(BaseModel):
    """Response after connecting units."""

    edge_id: str
    source_id: str
    target_id: str
    relation_type: str


# ── CognitiveMemory operations ──


class PrepareCallRequest(BaseModel):
    """Prepare context for an LLM call."""

    task_description: str
    task_type: str | None = None
    input_data: dict[str, Any] | None = None


class PrepareCallResponse(BaseModel):
    """Prepared context for an LLM call."""

    output_schema: dict[str, Any] | None = None
    system_prompt_fragment: str | None = None
    domain_context: list[dict[str, Any]]
    warnings: list[str]
    few_shot_examples: list[dict[str, Any]] | None = None
    estimated_tokens: int
    capacity_used: float


class RecordOutcomeRequest(BaseModel):
    """Record an episode after an LLM call."""

    task_description: str
    input_data: dict[str, Any] | None = None
    output: dict[str, Any]
    outcome: str = Field(description="success, failure, or corrected")
    correction: str | None = None
    task_type: str | None = None
    tags: list[str] = Field(default_factory=list)


class RecordOutcomeResponse(BaseModel):
    """Response after recording an episode."""

    episode_id: str


# ── Stats ──


class StatsResponse(BaseModel):
    """Memory system statistics."""

    semantic_nodes: int
    procedural_nodes: int
    episodic_nodes: int
    total_edges: int
    total_episodes: int
    pending_consolidation: int


# ── Consolidation ──


class ConsolidateResponse(BaseModel):
    """Consolidation results."""

    results_count: int
    details: list[dict[str, Any]]
