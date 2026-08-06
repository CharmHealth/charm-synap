"""synap MCP server — exposes agent memory tools via FastMCP.

Usage:
    # stdio (Claude Desktop, Cursor, etc.)
    python -m synap.mcp_server

    # HTTP
    python -m synap.mcp_server http

    # With custom database path
    SYNAP_DB=./my_memory python -m synap.mcp_server

Requires: pip install synap[kuzu] fastmcp
"""

from __future__ import annotations

import os
import sys
from typing import Any

from fastmcp import Context, FastMCP

mcp = FastMCP(name="synap Memory")

# Lazy-initialized memory instance (created on first tool call)
_memory = None


async def _get_memory():
    """Initialize CognitiveMemory on first use."""
    global _memory
    if _memory is not None:
        return _memory

    from synap.facade import CognitiveMemory
    from synap.graph import MemoryGraph
    from synap.semantic import SemanticMemory

    db_path = os.environ.get("SYNAP_DB")
    embedding_dim = int(os.environ.get("SYNAP_EMBEDDING_DIM", "8"))

    if db_path:
        from synap.backends.kuzu import KuzuBackend
        from synap.persistent_graph import PersistentGraph

        backend = KuzuBackend(db_path, embedding_dim=embedding_dim)
        graph = PersistentGraph(backend=backend)
    else:
        graph = MemoryGraph()

    # Offline placeholder providers — configure real ones for production
    # (wiring a real embedding model from an env var is tracked separately).
    from synap.providers import HashEmbedder, PlaceholderLLM

    embedder = HashEmbedder(dim=embedding_dim)
    llm = PlaceholderLLM()

    domain = SemanticMemory(
        graph=graph, embedding_provider=embedder, llm_provider=llm
    )
    _memory = CognitiveMemory(
        domain=domain,
        embedding_provider=embedder,
        llm_provider=llm,
        graph=graph,
    )
    return _memory


@mcp.tool
async def remember_fact(
    content: str,
    metadata: dict[str, Any] | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Store a fact in long-term semantic memory.

    <usecase>
    When the agent learns something worth remembering across conversations.
    Facts are stored with contradiction detection — if the new fact
    supersedes an existing one, the old fact is marked as outdated.
    </usecase>
    """
    memory = await _get_memory()
    node_id = await memory.domain.absorb(
        insights=[content],
        source_episodes=[],
        metadata={**(metadata or {}), "source": "mcp_tool"},
    )
    return {"stored": True, "node_id": node_id}


@mcp.tool
async def recall(
    query: str,
    memory_type: str = "all",
    limit: int = 5,
    ctx: Context = None,
) -> dict[str, Any]:
    """Search memory for relevant facts and past experiences.

    <usecase>
    When the agent needs context from prior knowledge or past interactions.
    Searches semantic memory (facts/knowledge) and episodic memory
    (past experiences with outcomes).
    </usecase>
    """
    memory = await _get_memory()
    results: dict[str, Any] = {}

    if memory_type in ("all", "semantic"):
        domain_results = await memory.domain.retrieve(query)
        results["facts"] = [
            {"content": r.content, "relevance": r.relevance}
            for r in domain_results[:limit]
        ]

    if memory_type in ("all", "episodic"):
        episodes = await memory.episodic.recall(query, max_episodes=limit)
        results["episodes"] = [
            {
                "cue": e.cue,
                "outcome": e.outcome.value,
                "task_type": e.task_type,
                "correction": e.correction,
            }
            for e in episodes
        ]

    return results


@mcp.tool
async def record_observation(
    description: str,
    outcome: str = "success",
    output: dict[str, Any] | None = None,
    correction: str | None = None,
    task_type: str | None = None,
    input_data: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Record an observation or experience for future learning.

    <usecase>
    After completing a task, record what happened so the system can
    learn from successes and failures. Append-only — cannot modify
    or delete past records.
    </usecase>
    """
    from synap.types import EpisodeOutcome

    memory = await _get_memory()
    episode_id = await memory.record_outcome(
        task_description=description,
        input_data=input_data,
        output=output or {},
        outcome=EpisodeOutcome(outcome),
        correction=correction,
        task_type=task_type,
        tags=tags or [],
    )
    return {"recorded": True, "episode_id": episode_id}


@mcp.tool
async def suggest_amendment(
    task_type: str,
    suggestion: str,
    rationale: str = "",
    ctx: Context = None,
) -> dict[str, Any]:
    """Propose a change to a task procedure.

    <usecase>
    When the agent identifies a missing step or check in a procedure.
    The suggestion is queued — it only takes effect if consolidation
    evidence supports it. Cannot directly modify procedures.
    </usecase>
    """
    from synap.types import MemoryNode, MemoryType

    memory = await _get_memory()
    node = MemoryNode(
        content=f"Proposed amendment for {task_type}: {suggestion}",
        node_type=MemoryType.SEMANTIC,
        metadata={
            "type": "amendment_proposal",
            "task_type": task_type,
            "suggestion": suggestion,
            "rationale": rationale,
            "source": "mcp_tool",
        },
    )
    await memory.graph.add_node(node)
    return {"queued": True, "proposal_id": node.id}


@mcp.tool
async def get_procedure(
    task_type: str | None = None,
    task_description: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Look up the active procedure and output schema for a task type.

    <usecase>
    When the agent needs to know the required reasoning steps for a task.
    Returns the field ordering and schema that enforce the procedure.
    </usecase>
    """
    memory = await _get_memory()
    proc = await memory.procedural.match(
        task_description or "", task_type=task_type
    )
    if proc is None:
        return {"found": False}
    return {
        "found": True,
        "task_type": proc.task_type,
        "description": proc.description,
        "field_ordering": proc.field_ordering,
        "schema": proc.schema,
    }


if __name__ == "__main__":
    use_stdio = True
    if len(sys.argv) > 1 and sys.argv[1].strip().lower() == "http":
        use_stdio = False

    if use_stdio:
        mcp.run()
    else:
        host = os.environ.get("SYNAP_HOST", "127.0.0.1")
        port = int(os.environ.get("SYNAP_PORT", "8080"))
        mcp.run(transport="http", host=host, port=port)
