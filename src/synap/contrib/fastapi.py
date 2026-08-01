"""FastAPI integration for synap.

Usage:
    from synap.contrib.fastapi import create_router

    router = create_router(memory)
    app.include_router(router, prefix="/api/memory")

Or use the convenience server:
    from synap.contrib.fastapi import create_app

    app = create_app(memory)
    # uvicorn app:app
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, HTTPException

from synap.contrib.models import (
    ConnectRequest,
    ConnectResponse,
    ConsolidateResponse,
    KnowledgeSource,
    KnowledgeUnitResponse,
    PrepareCallRequest,
    PrepareCallResponse,
    RecordOutcomeRequest,
    RecordOutcomeResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    StatsResponse,
    StoreKnowledgeRequest,
    StoreKnowledgeResponse,
)
from synap.facade import CognitiveMemory
from synap.semantic import SemanticMemory
from synap.types import EpisodeOutcome


def create_router(memory: CognitiveMemory) -> APIRouter:
    """Create a FastAPI router with all synap endpoints.

    Args:
        memory: A configured CognitiveMemory instance.

    Returns:
        APIRouter to include in your FastAPI app.
    """
    router = APIRouter(tags=["synap"])

    def _get_semantic() -> SemanticMemory:
        domain = memory.domain
        if not isinstance(domain, SemanticMemory):
            raise HTTPException(
                status_code=501,
                detail="Knowledge endpoints require SemanticMemory as the domain adapter",
            )
        return domain

    # ── Knowledge Unit endpoints ──

    @router.post("/knowledge", response_model=StoreKnowledgeResponse)
    async def store_knowledge(req: StoreKnowledgeRequest) -> StoreKnowledgeResponse:
        """Absorb a knowledge unit into the graph."""
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

        return StoreKnowledgeResponse(
            id=node_id, connections_created=len(req.connections)
        )

    @router.get("/knowledge/{node_id}", response_model=KnowledgeUnitResponse)
    async def get_knowledge(node_id: str) -> KnowledgeUnitResponse:
        """Get a specific knowledge unit by ID."""
        node = await memory.graph.get_node(node_id)
        if not node:
            raise HTTPException(status_code=404, detail="Knowledge unit not found")

        source_data = node.metadata.get("source", {})
        source = KnowledgeSource(
            app=source_data.get("app", "unknown"),
            document=source_data.get("document", "unknown"),
            author=source_data.get("author"),
            location=source_data.get("location"),
            document_type=source_data.get("document_type"),
        )

        outgoing = await memory.graph.traverse(node_id, max_depth=1, max_nodes=50)
        connected_ids = [n.id for n in outgoing if n.id != node_id]

        return KnowledgeUnitResponse(
            id=node.id,
            content=node.content,
            domain=node.metadata.get("domain", "unknown"),
            source=source,
            metadata={
                k: v
                for k, v in node.metadata.items()
                if k not in ("domain", "source")
            },
            created_at=node.created_at,
            connections=connected_ids,
        )

    @router.post("/knowledge/search", response_model=SearchResponse)
    async def search_knowledge(req: SearchRequest) -> SearchResponse:
        """Search for knowledge units by semantic similarity."""
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
                KnowledgeSource(**source_data)
                if source_data and isinstance(source_data, dict)
                else None
            )

            items.append(
                SearchResultItem(
                    id=node.id,
                    content=node.content,
                    relevance=node.utility_score,
                    metadata={
                        k: v
                        for k, v in node.metadata.items()
                        if k not in ("domain", "source")
                    },
                    domain=node.metadata.get("domain"),
                    source=source,
                )
            )

        return SearchResponse(results=items, query=req.query)

    @router.post(
        "/knowledge/{node_id}/connect", response_model=ConnectResponse
    )
    async def connect_knowledge(
        node_id: str, req: ConnectRequest
    ) -> ConnectResponse:
        """Link two knowledge units."""
        semantic = _get_semantic()

        source_node = await memory.graph.get_node(node_id)
        if not source_node:
            raise HTTPException(status_code=404, detail="Source node not found")

        target_node = await memory.graph.get_node(req.target_id)
        if not target_node:
            raise HTTPException(status_code=404, detail="Target node not found")

        edge_id = await semantic.link(
            source_id=node_id,
            target_id=req.target_id,
            relation_type=req.relation_type,
            weight=req.weight,
        )

        return ConnectResponse(
            edge_id=edge_id,
            source_id=node_id,
            target_id=req.target_id,
            relation_type=req.relation_type,
        )

    # ── CognitiveMemory endpoints ──

    @router.post("/prepare", response_model=PrepareCallResponse)
    async def prepare_call(req: PrepareCallRequest) -> PrepareCallResponse:
        """Prepare context for an LLM call."""
        ctx = await memory.prepare_call(
            task_description=req.task_description,
            task_type=req.task_type,
            input_data=req.input_data,
        )

        return PrepareCallResponse(
            output_schema=ctx.output_schema,
            system_prompt_fragment=ctx.system_prompt_fragment,
            domain_context=[asdict(dr) for dr in ctx.domain_context],
            warnings=ctx.warnings,
            few_shot_examples=ctx.few_shot_examples,
            estimated_tokens=ctx.estimated_tokens,
            capacity_used=ctx.capacity_used,
        )

    @router.post("/record", response_model=RecordOutcomeResponse)
    async def record_outcome(req: RecordOutcomeRequest) -> RecordOutcomeResponse:
        """Record an episode after an LLM call."""
        outcome_map = {
            "success": EpisodeOutcome.SUCCESS,
            "failure": EpisodeOutcome.FAILURE,
            "corrected": EpisodeOutcome.CORRECTED,
        }
        outcome = outcome_map.get(req.outcome.lower())
        if not outcome:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid outcome: {req.outcome}. Must be success, failure, or corrected.",
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

        return RecordOutcomeResponse(episode_id=episode_id)

    # ── Lifecycle endpoints ──

    @router.get("/stats", response_model=StatsResponse)
    async def get_stats() -> StatsResponse:
        """Get memory system statistics."""
        s = await memory.stats()
        return StatsResponse(
            semantic_nodes=s.semantic_nodes,
            procedural_nodes=s.procedural_nodes,
            episodic_nodes=s.episodic_nodes,
            total_edges=s.total_edges,
            total_episodes=s.total_episodes,
            pending_consolidation=s.pending_consolidation,
        )

    @router.post("/consolidate", response_model=ConsolidateResponse)
    async def consolidate() -> ConsolidateResponse:
        """Trigger consolidation pass."""
        results = await memory.consolidate()
        return ConsolidateResponse(
            results_count=len(results),
            details=[asdict(r) for r in results],
        )

    @router.get("/health")
    async def health() -> dict[str, str]:
        """Health check."""
        return {"status": "ok"}

    return router


def create_app(
    memory: CognitiveMemory,
    prefix: str = "/api/memory",
    **fastapi_kwargs: Any,
) -> Any:
    """Create a complete FastAPI app serving synap.

    Convenience for standalone deployment:
        app = create_app(memory)
        uvicorn.run(app, host="0.0.0.0", port=8100)
    """
    from fastapi import FastAPI

    app = FastAPI(title="Synap Memory Service", **fastapi_kwargs)
    router = create_router(memory)
    app.include_router(router, prefix=prefix)
    return app
