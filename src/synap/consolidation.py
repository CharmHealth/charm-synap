"""Consolidation engine — cross-subsystem memory evolution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from synap._utils import safe_parse_json
from synap.protocols import GraphStore, LLMProvider, SemanticDomain
from synap.types import (
    ConsolidationEvent,
    ConsolidationTrigger,
    Episode,
    EpisodeOutcome,
    MemoryNode,
    MemoryType,
    Procedure,
)

if TYPE_CHECKING:
    from synap.episodic import EpisodicMemory
    from synap.procedural import ProceduralMemory


def _consolidated_fact_id(metadata: dict[str, Any]) -> str | None:
    """Deterministic id for a consolidated fact, so re-consolidating the same
    pattern upserts one fact instead of spraying random-UUID duplicates.

    Identity is the stable pattern key (task_type + key), independent of how many
    episodes the pattern spans. Retrieval-triggered consolidations have no
    pattern; they key on the query string instead. Anything without either keeps
    a random id (no dedup).
    """
    task_type = metadata.get("task_type")
    pattern_key = metadata.get("pattern_key")
    if task_type and pattern_key:
        basis = f"pattern\x00{task_type}\x00{pattern_key}"
    elif metadata.get("query"):
        basis = f"query\x00{metadata['query']}"
    else:
        return None
    return "consolidated:" + hashlib.sha256(basis.encode()).hexdigest()[:24]


@dataclass
class ConsolidationConfig:
    min_pattern_occurrences: int = 3
    eviction_threshold: float = 0.1
    max_queue_size: int = 100


@dataclass
class ConsolidationResult:
    event: ConsolidationEvent
    domain_id: str | None = None
    success: bool = True
    error: str | None = None


class ConsolidationEngine:
    """Processes consolidation events: episodic → domain knowledge."""

    def __init__(
        self,
        graph: GraphStore,
        domain: SemanticDomain,
        procedural: ProceduralMemory,
        episodic: EpisodicMemory,
        llm_provider: LLMProvider,
        config: ConsolidationConfig | None = None,
    ) -> None:
        self._graph = graph
        self._domain = domain
        self._procedural = procedural
        self._episodic = episodic
        self._llm = llm_provider
        self._config = config or ConsolidationConfig()
        self._queue: list[ConsolidationEvent] = []
        self._processed_patterns: set[tuple[str | None, str | None]] = set()

    async def on_episode_recorded(self, episode: Episode) -> ConsolidationEvent | None:
        if episode.task_type is None:
            return None

        patterns = await self._episodic.find_patterns(
            task_type=episode.task_type,
            min_occurrences=self._config.min_pattern_occurrences,
        )

        for pattern in patterns:
            if pattern.outcome == EpisodeOutcome.FAILURE:
                candidate_nodes = [
                    node
                    for eid in pattern.episode_ids
                    if (node := await self._graph.get_node(f"{eid}_content")) is not None
                ]
                if not candidate_nodes:
                    continue
                return ConsolidationEvent(
                    source_type=MemoryType.EPISODIC,
                    target_type=MemoryType.PROCEDURAL,
                    candidates=candidate_nodes,
                    trigger=ConsolidationTrigger.EVENT,
                    confidence=min(1.0, pattern.occurrences / 5),
                    metadata={
                        "task_type": pattern.task_type,
                        "pattern": pattern.pattern_description,
                        "pattern_key": pattern.key,
                    },
                )
            elif pattern.outcome == EpisodeOutcome.SUCCESS:
                candidate_nodes = [
                    node
                    for eid in pattern.episode_ids
                    if (node := await self._graph.get_node(f"{eid}_content")) is not None
                ]
                if candidate_nodes:
                    return ConsolidationEvent(
                        source_type=MemoryType.EPISODIC,
                        target_type=MemoryType.SEMANTIC,
                        candidates=candidate_nodes,
                        trigger=ConsolidationTrigger.EVENT,
                        confidence=min(1.0, pattern.occurrences / 5),
                        metadata={
                            "task_type": pattern.task_type,
                            "pattern": pattern.pattern_description,
                            "pattern_key": pattern.key,
                        },
                    )

        return None

    async def run_periodic(self) -> list[ConsolidationEvent]:
        await self._graph.decay_all()
        await self._graph.evict(threshold=self._config.eviction_threshold)

        events: list[ConsolidationEvent] = []
        seen_task_types: set[str] = set()

        # Skip patterns already processed from event-driven consolidation
        queued_patterns = self._processed_patterns | {
            (e.metadata.get("task_type"), e.metadata.get("pattern_key"))
            for e in self._queue
        }

        all_episodes = await self._episodic.all_episodes()
        for episode in all_episodes:
            if episode.task_type and episode.task_type not in seen_task_types:
                seen_task_types.add(episode.task_type)
                patterns = await self._episodic.find_patterns(
                    task_type=episode.task_type,
                    min_occurrences=self._config.min_pattern_occurrences,
                )
                for pattern in patterns:
                    pattern_key = (pattern.task_type, pattern.key)
                    if pattern_key in queued_patterns:
                        continue
                    candidate_nodes = [
                        node
                        for eid in pattern.episode_ids
                        if (node := await self._graph.get_node(f"{eid}_content"))
                        is not None
                    ]
                    if candidate_nodes:
                        target = (
                            MemoryType.PROCEDURAL
                            if pattern.outcome == EpisodeOutcome.FAILURE
                            else MemoryType.SEMANTIC
                        )
                        events.append(
                            ConsolidationEvent(
                                source_type=MemoryType.EPISODIC,
                                target_type=target,
                                candidates=candidate_nodes,
                                trigger=ConsolidationTrigger.PERIODIC,
                                confidence=min(1.0, pattern.occurrences / 5),
                                metadata={
                                    "task_type": pattern.task_type,
                                    "pattern": pattern.pattern_description,
                                    "pattern_key": pattern.key,
                                },
                            )
                        )

        return events

    async def on_retrieval(
        self,
        query: str,
        results: list[MemoryNode],
    ) -> ConsolidationEvent | None:
        episodic_results = [
            r for r in results if r.node_type == MemoryType.EPISODIC
        ]
        if len(episodic_results) < self._config.min_pattern_occurrences:
            return None

        return ConsolidationEvent(
            source_type=MemoryType.EPISODIC,
            target_type=MemoryType.SEMANTIC,
            candidates=episodic_results,
            trigger=ConsolidationTrigger.QUERY,
            confidence=0.5,
            metadata={"query": query},
        )

    def snapshot_queued_patterns(self) -> None:
        """Capture current queue patterns before draining."""
        self._processed_patterns = {
            (e.metadata.get("task_type"), e.metadata.get("pattern_key"))
            for e in self._queue
        }

    def clear_pattern_snapshot(self) -> None:
        """Clear the snapshot after consolidation is complete."""
        self._processed_patterns.clear()

    def queue_event(self, event: ConsolidationEvent) -> None:
        if len(self._queue) < self._config.max_queue_size:
            self._queue.append(event)

    async def process_queue(self) -> list[ConsolidationResult]:
        results = []
        while self._queue:
            event = self._queue.pop(0)
            result = await self.process(event)
            if result:
                results.append(result)
        return results

    async def process(self, event: ConsolidationEvent) -> ConsolidationResult | None:
        if not event.candidates:
            return None

        try:
            if (
                event.source_type == MemoryType.EPISODIC
                and event.target_type == MemoryType.SEMANTIC
            ):
                return await self._consolidate_to_semantic(event)
            elif (
                event.source_type == MemoryType.EPISODIC
                and event.target_type == MemoryType.PROCEDURAL
            ):
                return await self._consolidate_to_procedural(event)
            elif (
                event.source_type == MemoryType.SEMANTIC
                and event.target_type == MemoryType.SEMANTIC
            ):
                return await self._merge_semantic(event)
            else:
                return ConsolidationResult(
                    event=event,
                    success=False,
                    error=f"Unsupported consolidation: {event.source_type} → {event.target_type}",
                )
        except Exception as e:
            return ConsolidationResult(
                event=event,
                success=False,
                error=str(e),
            )

    async def _consolidate_to_semantic(
        self, event: ConsolidationEvent
    ) -> ConsolidationResult:
        contents = [c.content for c in event.candidates]
        prompt = (
            "Extract the key facts from these agent experiences into a single, "
            "concise statement. Focus on what is consistently true across all "
            "experiences, not the details of individual episodes.\n\n"
            "Experiences:\n"
            + "\n---\n".join(contents)
            + "\n\nExtracted fact:"
        )
        fact = await self._llm.generate(prompt)

        domain_id = await self._domain.absorb(
            insights=[fact.strip()],
            source_episodes=event.candidates,
            metadata=event.metadata,
            node_id=_consolidated_fact_id(event.metadata),
        )

        for candidate in event.candidates:
            candidate.utility_score *= 0.5

        return ConsolidationResult(
            event=event,
            domain_id=domain_id,
        )

    async def _consolidate_to_procedural(
        self, event: ConsolidationEvent
    ) -> ConsolidationResult:
        contents = [c.content for c in event.candidates]
        task_type = event.metadata.get("task_type", "unknown")

        # Find the existing procedure to amend
        existing = await self._procedural.match(task_type)
        if existing is None:
            # No procedure to amend — fall back to semantic storage
            return await self._consolidate_to_semantic(event)

        # Don't re-amend for a pattern this procedure already absorbed. Repeated
        # consolidation of the same failure pattern would otherwise churn a new
        # version every cycle and re-insert the same verification field.
        pattern_key = event.metadata.get("pattern_key")
        amended_by = list(existing.metadata.get("amended_by", []))
        if pattern_key and pattern_key in amended_by:
            return ConsolidationResult(event=event, domain_id=existing.id, success=True)

        prompt = (
            f"The following procedure has a repeated failure pattern.\n\n"
            f"Task type: {existing.task_type}\n"
            f"Description: {existing.description}\n"
            f"Current fields (in reasoning order): {existing.field_ordering}\n"
            f"Current schema:\n{json.dumps(existing.schema, indent=2)}\n\n"
            f"Failure pattern: {event.metadata.get('pattern', '')}\n\n"
            f"Failure episodes:\n"
            + "\n---\n".join(contents)
            + "\n\nPropose ONE new check or reasoning field that would prevent "
            "this failure. The field should enforce a verification step the model "
            "is currently skipping.\n\n"
            "Respond with JSON:\n"
            '{"field_name": "snake_case_name", '
            '"field_type": "string", '
            '"field_description": "What the model must provide in this field", '
            '"insert_before": "existing_field_name"}\n\n'
            "The new field will be required before the model can fill in the "
            "insert_before field."
        )
        raw = await self._llm.generate(prompt)
        amendment = safe_parse_json(raw)

        if amendment is None or "field_name" not in amendment:
            # LLM failed to produce valid structured amendment — fall back
            return await self._consolidate_to_semantic(event)

        field_name = amendment["field_name"]
        insert_before = amendment.get("insert_before")

        # Build amended schema
        new_schema = dict(existing.schema)
        new_schema[field_name] = {
            "type": amendment.get("field_type", "string"),
            "description": amendment.get("field_description", ""),
        }

        # Build amended field ordering
        new_ordering = list(existing.field_ordering)
        if insert_before and insert_before in new_ordering:
            idx = new_ordering.index(insert_before)
            new_ordering.insert(idx, field_name)
        else:
            new_ordering.insert(max(0, len(new_ordering) - 1), field_name)

        # Build amended prerequisites
        new_prereqs = {k: list(v) for k, v in existing.prerequisite_fields.items()}
        if insert_before and insert_before in new_prereqs:
            new_prereqs[insert_before].append(field_name)
        elif insert_before:
            new_prereqs[insert_before] = [field_name]

        # Register new procedure version (creates supersedes edge)
        new_procedure = Procedure(
            task_type=existing.task_type,
            description=existing.description,
            schema=new_schema,
            field_ordering=new_ordering,
            prerequisite_fields=new_prereqs,
            system_prompt_fragment=existing.system_prompt_fragment,
            metadata={
                **existing.metadata,
                "amendment_source": "consolidation",
                "pattern": event.metadata.get("pattern", ""),
                "amended_by": amended_by + ([pattern_key] if pattern_key else []),
            },
            episode_ids=[c.metadata.get("episode_id", c.id) for c in event.candidates],
        )
        proc_id = await self._procedural.register(new_procedure)

        return ConsolidationResult(
            event=event,
            domain_id=proc_id,
        )

    async def _merge_semantic(self, event: ConsolidationEvent) -> ConsolidationResult:
        contents = [c.content for c in event.candidates]
        prompt = (
            "These semantic facts are redundant. Merge them into a single, "
            "comprehensive statement that preserves all unique information:\n\n"
            + "\n---\n".join(contents)
            + "\n\nMerged fact:"
        )
        merged = await self._llm.generate(prompt)

        meta = dict(event.metadata)
        meta["merged_from"] = [c.id for c in event.candidates]

        domain_id = await self._domain.absorb(
            insights=[merged.strip()],
            source_episodes=event.candidates,
            metadata=meta,
        )

        for candidate in event.candidates:
            candidate.utility_score *= 0.3

        return ConsolidationResult(
            event=event,
            domain_id=domain_id,
        )
