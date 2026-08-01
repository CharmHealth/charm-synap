"""Core types for the synap memory system."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class MemoryType(Enum):
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    EPISODIC = "episodic"


class EpisodeOutcome(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    CORRECTED = "corrected"


class ConsolidationTrigger(Enum):
    EVENT = "event"
    PERIODIC = "periodic"
    QUERY = "query"


def _now() -> datetime:
    return datetime.now(UTC)


def _uuid() -> str:
    return uuid.uuid4().hex


@dataclass
class MemoryNode:
    """Atomic unit of storage across all subsystems."""

    content: str
    node_type: MemoryType
    id: str = field(default_factory=_uuid)
    created_at: datetime = field(default_factory=_now)
    last_accessed: datetime = field(default_factory=_now)
    access_count: int = 0
    utility_score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    def touch(self) -> None:
        """Record an access — updates recency and count."""
        self.last_accessed = _now()
        self.access_count += 1


@dataclass
class MemoryEdge:
    """Typed relation between two nodes, within or across subsystems."""

    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    id: str = field(default_factory=_uuid)
    created_at: datetime = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CapacityHints:
    """Model-agnostic configuration for retrieval depth and budget."""

    max_context_tokens: int = 8192
    recommended_chunk_tokens: int = 3000
    quantization_tier: str | None = None
    reserved_tokens: int = 1500


@dataclass
class Procedure:
    """A decision procedure that maps task types to enforced output schemas."""

    task_type: str
    description: str
    schema: dict[str, Any]
    field_ordering: list[str]
    prerequisite_fields: dict[str, list[str]] = field(default_factory=dict)
    system_prompt_fragment: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_uuid)
    episode_ids: list[str] = field(default_factory=list)


@dataclass
class ToolCall:
    """A single tool invocation within an episode, capturing the full context."""

    query: str              # what the agent was trying to accomplish
    server: str             # MCP server (e.g. "news-mcp")
    tool_name: str          # specific tool (e.g. "search_articles")
    parameters: dict[str, Any]  # what was passed
    result_summary: str     # truncated result
    success: bool


@dataclass
class Episode:
    """A recorded agent experience with outcome."""

    cue: str
    content: dict[str, Any]
    outcome: EpisodeOutcome
    correction: str | None = None
    task_type: str | None = None
    input_data: dict[str, Any] | None = None
    id: str = field(default_factory=_uuid)
    timestamp: datetime = field(default_factory=_now)
    tags: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class ConsolidationEvent:
    """Cross-subsystem consolidation request."""

    source_type: MemoryType
    target_type: MemoryType
    candidates: list[MemoryNode]
    trigger: ConsolidationTrigger
    confidence: float = 0.0
    suggested_content: str | None = None
    id: str = field(default_factory=_uuid)
    created_at: datetime = field(default_factory=_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainResult:
    """A piece of domain knowledge surfaced by a SemanticDomain adapter."""

    content: str
    relevance: float = 1.0
    source_id: str = field(default_factory=_uuid)
    metadata: dict[str, Any] = field(default_factory=dict)
    valid_from: datetime | None = None
    valid_until: datetime | None = None


@dataclass
class PreparedContext:
    """Everything needed to build an LLM call, returned by CognitiveMemory.prepare_call()."""

    # Procedural
    procedure: Procedure | None = None
    output_schema: dict[str, Any] | None = None
    system_prompt_fragment: str | None = None

    # Domain knowledge
    domain_context: list[DomainResult] = field(default_factory=list)

    # Episodic
    relevant_episodes: list[Episode] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    few_shot_examples: list[dict[str, Any]] | None = None

    # Budget metadata
    estimated_tokens: int = 0
    capacity_used: float = 0.0
