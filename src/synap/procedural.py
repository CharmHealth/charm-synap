"""Procedural memory — task-type to output schema registry with enforcement."""

from __future__ import annotations

from typing import Any

from synap.protocols import EmbeddingProvider, GraphStore
from synap.types import MemoryEdge, MemoryNode, MemoryType, Procedure


class ProceduralMemory:
    """Maps task types to output schemas that enforce procedures.

    The key mechanism: procedural memory doesn't inject instructions
    into the prompt. It produces an output schema where field ordering
    IS the procedure.
    """

    def __init__(
        self,
        graph: GraphStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._graph = graph
        self._embedder = embedding_provider
        self._procedures: dict[str, Procedure] = {}
        self._task_type_index: dict[str, str] = {}

    async def register(self, procedure: Procedure) -> str:
        existing_id = self._task_type_index.get(procedure.task_type)

        node = MemoryNode(
            content=f"{procedure.task_type}: {procedure.description}",
            node_type=MemoryType.PROCEDURAL,
            id=procedure.id,
            embedding=await self._embedder.embed(
                f"{procedure.task_type} {procedure.description}"
            ),
            metadata={
                "task_type": procedure.task_type,
                "description": procedure.description,
                "schema": procedure.schema,
                "field_ordering": procedure.field_ordering,
                "prerequisite_fields": procedure.prerequisite_fields,
                "system_prompt_fragment": procedure.system_prompt_fragment,
                "procedure_metadata": procedure.metadata,
                "episode_ids": procedure.episode_ids,
            },
        )
        nodes = [node]
        edges: list[MemoryEdge] = []
        if existing_id and existing_id != procedure.id:
            old_node = await self._graph.get_node(existing_id)
            if old_node is not None:
                # Tombstone the retired version intrinsically (status), and keep
                # the supersedes edge for lineage. Status reads consult the flag,
                # never the edge, so deleting this new version cannot resurrect
                # the old one.
                old_node.metadata = {**old_node.metadata, "superseded": True}
                nodes.append(old_node)
                edges.append(
                    MemoryEdge(
                        source_id=procedure.id,
                        target_id=existing_id,
                        relation_type="supersedes",
                    )
                )

        # One atomic write: the new version, the retired flag on the old version,
        # and the lineage edge land together or not at all.
        await self._graph.write_batch(nodes, edges)

        self._procedures[procedure.id] = procedure
        self._task_type_index[procedure.task_type] = procedure.id

        return procedure.id

    async def match(
        self,
        task_description: str,
        task_type: str | None = None,
    ) -> Procedure | None:
        if await self._graph.node_count(MemoryType.PROCEDURAL) == 0:
            return None

        # Exact task_type match (highest priority)
        if task_type:
            nodes = await self._graph.query(
                node_type=MemoryType.PROCEDURAL,
                filters={"task_type": task_type},
                limit=10,
            )
            for node in nodes:
                proc = await self._reconstruct_procedure(node)
                if proc and await self._is_active(node):
                    return proc

        # Structural match: task_type substring in description
        nodes = await self._graph.query(
            node_type=MemoryType.PROCEDURAL, limit=100
        )
        for node in nodes:
            node_task_type = node.metadata.get("task_type", "")
            if node_task_type and node_task_type in task_description:
                proc = await self._reconstruct_procedure(node)
                if proc and await self._is_active(node):
                    return proc

        # Fallback: similarity search
        query_embedding = await self._embedder.embed(task_description)
        similar = await self._graph.similarity_search(
            query_embedding, node_type=MemoryType.PROCEDURAL, limit=5
        )

        for node in similar:
            proc = await self._reconstruct_procedure(node)
            if proc and await self._is_active(node):
                return proc

        return None

    async def build_schema(
        self,
        procedure: Procedure,
        episode_context: list[MemoryNode] | None = None,
    ) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        required: list[str] = []

        for field_name in procedure.field_ordering:
            field_schema = procedure.schema.get(field_name, {"type": "string"})
            field_def = dict(field_schema)

            if episode_context and field_name in procedure.prerequisite_fields:
                hints = self._corrective_hints(episode_context)
                if hints:
                    existing_desc = field_def.get("description", "")
                    field_def["description"] = (
                        f"{existing_desc} WARNING: {hints}".strip()
                    )

            properties[field_name] = field_def
            required.append(field_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    async def get_procedure(self, procedure_id: str) -> Procedure | None:
        if procedure_id in self._procedures:
            return self._procedures[procedure_id]
        node = await self._graph.get_node(procedure_id)
        if node is None or node.node_type != MemoryType.PROCEDURAL:
            return None
        return await self._reconstruct_procedure(node)

    async def list_procedures(self, active_only: bool = True) -> list[Procedure]:
        nodes = await self._graph.query(node_type=MemoryType.PROCEDURAL)
        procedures = []
        for node in nodes:
            proc = await self._reconstruct_procedure(node)
            if proc is None:
                continue
            if active_only and not await self._is_active(node):
                continue
            procedures.append(proc)
        return procedures

    async def _reconstruct_procedure(self, node: MemoryNode) -> Procedure | None:
        """Reconstruct a Procedure from its graph node metadata."""
        if node.id in self._procedures:
            return self._procedures[node.id]

        meta = node.metadata
        task_type = meta.get("task_type")
        if task_type is None:
            return None

        procedure = Procedure(
            id=node.id,
            task_type=task_type,
            description=meta.get("description", ""),
            schema=meta.get("schema", {}),
            field_ordering=meta.get("field_ordering", []),
            prerequisite_fields=meta.get("prerequisite_fields", {}),
            system_prompt_fragment=meta.get("system_prompt_fragment"),
            metadata=meta.get("procedure_metadata", {}),
            episode_ids=meta.get("episode_ids", []),
        )

        self._procedures[node.id] = procedure
        self._task_type_index[task_type] = node.id
        return procedure

    async def _is_active(self, node: MemoryNode) -> bool:
        """Active unless intrinsically tombstoned. Reads the `superseded` flag,
        never the supersedes edge, so deletion of a superseder can't reactivate
        a retired version."""
        return not node.metadata.get("superseded", False)

    def _corrective_hints(self, episodes: list[MemoryNode]) -> str:
        """Extract correction text from failure/corrected outcome nodes."""
        hints = []
        for ep in episodes:
            outcome = ep.metadata.get("outcome")
            correction = ep.metadata.get("correction")
            if outcome in ("failure", "corrected") and correction:
                hints.append(correction)
        return "; ".join(hints) if hints else ""
