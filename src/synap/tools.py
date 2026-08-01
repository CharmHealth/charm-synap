"""Agent-callable memory tools with per-type permission model.

Semantic: read + write (with contradiction detection)
Procedural: read + suggest amendments (queued, not direct)
Episodic: read + append observations (no delete, no modify)

Usage:
    from synap.tools import memory_tools
    tools = memory_tools(cognitive_memory)
    # Returns list of {"name": str, "schema": dict, "handler": async_fn}
"""

from __future__ import annotations

from typing import Any

from synap.facade import CognitiveMemory
from synap.types import EpisodeOutcome, MemoryNode, MemoryType


def memory_tools(memory: CognitiveMemory) -> list[dict[str, Any]]:
    """Create agent-callable tool definitions wrapping a CognitiveMemory instance.

    Returns a list of tool dicts, each with:
    - name: tool identifier
    - description: what the tool does
    - schema: JSON Schema for parameters
    - handler: async function(params) -> result
    """

    async def remember_fact(params: dict[str, Any]) -> dict[str, Any]:
        """Store a fact in semantic memory with contradiction detection."""
        content = params["content"]
        metadata = params.get("metadata", {})
        node_id = await memory.domain.absorb(
            insights=[content],
            source_episodes=[],
            metadata={**metadata, "source": "agent_tool"},
        )
        return {"stored": True, "node_id": node_id}

    async def recall(params: dict[str, Any]) -> dict[str, Any]:
        """Search across semantic and episodic memory."""
        query = params["query"]
        memory_type = params.get("memory_type", "all")
        limit = params.get("limit", 5)

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

    async def record_observation(params: dict[str, Any]) -> dict[str, Any]:
        """Record an agent observation as an episodic experience (append-only)."""
        episode_id = await memory.record_outcome(
            task_description=params["description"],
            input_data=params.get("input_data"),
            output=params.get("output", {}),
            outcome=EpisodeOutcome(params.get("outcome", "success")),
            correction=params.get("correction"),
            task_type=params.get("task_type"),
            tags=params.get("tags", []),
        )
        return {"recorded": True, "episode_id": episode_id}

    async def suggest_amendment(params: dict[str, Any]) -> dict[str, Any]:
        """Propose a procedural amendment (queued for consolidation, not immediate)."""
        task_type = params["task_type"]
        suggestion = params["suggestion"]
        rationale = params.get("rationale", "")

        # Store as a proposal node in the graph
        node = MemoryNode(
            content=f"Proposed amendment for {task_type}: {suggestion}",
            node_type=MemoryType.SEMANTIC,
            metadata={
                "type": "amendment_proposal",
                "task_type": task_type,
                "suggestion": suggestion,
                "rationale": rationale,
                "source": "agent_tool",
            },
        )
        await memory.graph.add_node(node)
        return {"queued": True, "proposal_id": node.id}

    async def get_procedure(params: dict[str, Any]) -> dict[str, Any]:
        """Look up the active procedure for a task type."""
        proc = await memory.procedural.match(
            params.get("task_description", ""),
            task_type=params.get("task_type"),
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

    return [
        {
            "name": "remember_fact",
            "description": "Store a fact in long-term semantic memory. Use when the agent learns something worth remembering across conversations.",
            "schema": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The fact to remember"},
                    "metadata": {"type": "object", "description": "Optional metadata tags"},
                },
                "required": ["content"],
            },
            "handler": remember_fact,
        },
        {
            "name": "recall",
            "description": "Search memory for relevant facts and past experiences.",
            "schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "memory_type": {
                        "type": "string",
                        "enum": ["all", "semantic", "episodic"],
                        "description": "Which memory to search (default: all)",
                    },
                    "limit": {"type": "integer", "description": "Max results (default: 5)"},
                },
                "required": ["query"],
            },
            "handler": recall,
        },
        {
            "name": "record_observation",
            "description": "Record an observation or experience for future learning. Append-only — cannot modify or delete past records.",
            "schema": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "What happened"},
                    "output": {"type": "object", "description": "The agent's output/result"},
                    "outcome": {
                        "type": "string",
                        "enum": ["success", "failure", "corrected"],
                        "description": "How it went",
                    },
                    "correction": {"type": "string", "description": "What should have happened (if corrected)"},
                    "task_type": {"type": "string", "description": "Task type identifier"},
                    "input_data": {"type": "object", "description": "Input that triggered this"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["description"],
            },
            "handler": record_observation,
        },
        {
            "name": "suggest_amendment",
            "description": "Propose a change to a task procedure. The suggestion is queued and only takes effect if consolidation evidence supports it. Cannot directly modify procedures.",
            "schema": {
                "type": "object",
                "properties": {
                    "task_type": {"type": "string", "description": "Which procedure to amend"},
                    "suggestion": {"type": "string", "description": "The proposed change"},
                    "rationale": {"type": "string", "description": "Why this change would help"},
                },
                "required": ["task_type", "suggestion"],
            },
            "handler": suggest_amendment,
        },
        {
            "name": "get_procedure",
            "description": "Look up the active procedure and output schema for a task type.",
            "schema": {
                "type": "object",
                "properties": {
                    "task_type": {"type": "string", "description": "Task type to look up"},
                    "task_description": {"type": "string", "description": "Description to match against"},
                },
            },
            "handler": get_procedure,
        },
    ]
