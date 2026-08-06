"""Sanic integration for synap.

Usage:
    from synap.contrib.sanic import create_blueprint

    bp = create_blueprint(memory)
    app.blueprint(bp, url_prefix="/api/memory")

Or use the convenience server:
    from synap.contrib.sanic import create_app

    app = create_app(memory)
    app.run(host="0.0.0.0", port=8100)
"""

from __future__ import annotations

import datetime
from dataclasses import asdict
from typing import Any

from sanic import Blueprint, HTTPResponse
from sanic import json as sanic_json
from sanic.exceptions import InvalidUsage, NotFound

from synap.contrib.models import (
    ConnectRequest,
    PrepareCallRequest,
    RecordOutcomeRequest,
    SearchRequest,
    StoreKnowledgeRequest,
)
from synap.facade import CognitiveMemory
from synap.semantic import SemanticMemory
from synap.types import EpisodeOutcome


def _serialize(obj: Any) -> Any:
    """Make an object JSON-serializable (Sanic uses ujson, no datetime support)."""
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def _json(data: dict[str, Any] | list[Any], status: int = 200) -> HTTPResponse:
    return sanic_json(_serialize(data), status=status)


def _parse(model_cls: type, body: dict[str, Any]) -> Any:
    try:
        return model_cls.model_validate(body)  # type: ignore[attr-defined]  # pydantic BaseModel subclass
    except Exception as e:
        raise InvalidUsage(f"Invalid request: {e}") from e


def create_blueprint(memory: CognitiveMemory) -> Blueprint:
    """Create a Sanic blueprint with all synap endpoints.

    Args:
        memory: A configured CognitiveMemory instance.

    Returns:
        Blueprint to register on your Sanic app.
    """
    bp = Blueprint("synap", url_prefix="")

    def _get_semantic() -> SemanticMemory:
        domain = memory.domain
        if not isinstance(domain, SemanticMemory):
            raise InvalidUsage(
                "Knowledge endpoints require SemanticMemory as the domain adapter"
            )
        return domain

    # ── Knowledge Unit endpoints ──

    @bp.post("/knowledge")
    async def store_knowledge(request) -> HTTPResponse:
        """Absorb a knowledge unit into the graph."""
        req = _parse(StoreKnowledgeRequest, request.json)
        semantic = _get_semantic()

        metadata = dict(req.metadata)
        metadata["domain"] = req.domain
        metadata["source"] = req.source.model_dump(exclude_none=True)

        relations = None
        if req.connections:
            relations = [
                ("__self__", "related_to", conn_id) for conn_id in req.connections
            ]

        node_id = await semantic.store(
            content=req.content,
            metadata=metadata,
            relations=relations,
            check_contradictions=True,
        )

        return _json({"id": node_id, "connections_created": len(req.connections)})

    @bp.get("/knowledge/<node_id:str>")
    async def get_knowledge(request, node_id: str) -> HTTPResponse:
        """Get a specific knowledge unit by ID."""
        node = await memory.graph.get_node(node_id)
        if not node:
            raise NotFound("Knowledge unit not found")

        source_data = node.metadata.get("source", {})
        source = {
            "app": source_data.get("app", "unknown"),
            "document": source_data.get("document", "unknown"),
            "author": source_data.get("author"),
            "location": source_data.get("location"),
            "document_type": source_data.get("document_type"),
        }

        outgoing = await memory.graph.traverse(node_id, max_depth=1, max_nodes=50)
        connected_ids = [n.id for n in outgoing if n.id != node_id]

        return _json({
            "id": node.id,
            "content": node.content,
            "domain": node.metadata.get("domain", "unknown"),
            "source": {k: v for k, v in source.items() if v is not None},
            "metadata": {
                k: v for k, v in node.metadata.items() if k not in ("domain", "source")
            },
            "created_at": node.created_at.isoformat(),
            "connections": connected_ids,
        })

    @bp.post("/knowledge/search")
    async def search_knowledge(request) -> HTTPResponse:
        """Search for knowledge units by semantic similarity."""
        req = _parse(SearchRequest, request.json)
        semantic = _get_semantic()

        result = await semantic.search(
            query=req.query,
            max_depth=req.max_depth,
            max_nodes=req.max_results,
        )

        items = []
        for node in result.nodes:
            if req.domain and node.metadata.get("domain") != req.domain:
                continue

            source_data = node.metadata.get("source")
            source = (
                {k: v for k, v in source_data.items() if v is not None}
                if source_data and isinstance(source_data, dict)
                else None
            )

            items.append({
                "id": node.id,
                "content": node.content,
                "relevance": node.utility_score,
                "metadata": {
                    k: v for k, v in node.metadata.items() if k not in ("domain", "source")
                },
                "domain": node.metadata.get("domain"),
                "source": source,
            })

        return _json({"results": items, "query": req.query})

    @bp.post("/knowledge/<node_id:str>/connect")
    async def connect_knowledge(request, node_id: str) -> HTTPResponse:
        """Link two knowledge units."""
        req = _parse(ConnectRequest, request.json)
        semantic = _get_semantic()

        source_node = await memory.graph.get_node(node_id)
        if not source_node:
            raise NotFound("Source node not found")

        target_node = await memory.graph.get_node(req.target_id)
        if not target_node:
            raise NotFound("Target node not found")

        edge_id = await semantic.link(
            source_id=node_id,
            target_id=req.target_id,
            relation_type=req.relation_type,
            weight=req.weight,
        )

        return _json({
            "edge_id": edge_id,
            "source_id": node_id,
            "target_id": req.target_id,
            "relation_type": req.relation_type,
        })

    # ── CognitiveMemory endpoints ──

    @bp.post("/prepare")
    async def prepare_call(request) -> HTTPResponse:
        """Prepare context for an LLM call."""
        req = _parse(PrepareCallRequest, request.json)

        ctx = await memory.prepare_call(
            task_description=req.task_description,
            task_type=req.task_type,
            input_data=req.input_data,
        )

        return _json({
            "output_schema": ctx.output_schema,
            "system_prompt_fragment": ctx.system_prompt_fragment,
            "domain_context": [asdict(dr) for dr in ctx.domain_context],
            "warnings": ctx.warnings,
            "few_shot_examples": ctx.few_shot_examples,
            "estimated_tokens": ctx.estimated_tokens,
            "capacity_used": ctx.capacity_used,
        })

    @bp.post("/record")
    async def record_outcome(request) -> HTTPResponse:
        """Record an episode after an LLM call."""
        req = _parse(RecordOutcomeRequest, request.json)

        outcome_map = {
            "success": EpisodeOutcome.SUCCESS,
            "failure": EpisodeOutcome.FAILURE,
            "corrected": EpisodeOutcome.CORRECTED,
        }
        outcome = outcome_map.get(req.outcome.lower())
        if not outcome:
            raise InvalidUsage(
                f"Invalid outcome: {req.outcome}. Must be success, failure, or corrected."
            )

        episode_id = await memory.record_outcome(
            task_description=req.task_description,
            input_data=req.input_data,
            output=req.output,
            outcome=outcome,
            correction=req.correction,
            task_type=req.task_type,
            tags=req.tags,
        )

        return _json({"episode_id": episode_id})

    # ── Lifecycle endpoints ──

    @bp.get("/stats")
    async def get_stats(request) -> HTTPResponse:
        """Get memory system statistics."""
        s = await memory.stats()
        return _json({
            "semantic_nodes": s.semantic_nodes,
            "procedural_nodes": s.procedural_nodes,
            "episodic_nodes": s.episodic_nodes,
            "total_edges": s.total_edges,
            "total_episodes": s.total_episodes,
            "pending_consolidation": s.pending_consolidation,
        })

    @bp.post("/consolidate")
    async def consolidate(request) -> HTTPResponse:
        """Trigger consolidation pass."""
        results = await memory.consolidate()
        return _json({
            "results_count": len(results),
            "details": [asdict(r) for r in results],
        })

    @bp.get("/health")
    async def health(request) -> HTTPResponse:
        """Health check."""
        return _json({"status": "ok"})

    return bp


def create_app(
    memory: CognitiveMemory,
    prefix: str = "/api/memory",
    **sanic_kwargs: Any,
) -> Any:
    """Create a complete Sanic app serving synap.

    Convenience for standalone deployment:
        app = create_app(memory)
        app.run(host="0.0.0.0", port=8100)
    """
    from sanic import Sanic

    app = Sanic("synap", **sanic_kwargs)
    bp = create_blueprint(memory)
    app.blueprint(bp, url_prefix=prefix)
    return app
