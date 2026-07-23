"""Episodic memory — cue-tag-content graph with reconstructive retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from synap._utils import cosine_similarity
from synap.protocols import EmbeddingProvider, GraphStore
from synap.types import (
    CapacityHints,
    Episode,
    EpisodeOutcome,
    MemoryEdge,
    MemoryNode,
    MemoryType,
    ToolCall,
)


@dataclass
class EpisodicPattern:
    """A repeated pattern detected across episodes."""

    task_type: str
    pattern_description: str
    occurrences: int
    episode_ids: list[str]
    outcome: EpisodeOutcome


class EpisodicMemory:
    """Past agent experiences with reconstructive retrieval.

    Each episode is stored as a small subgraph:
    - Cue node: what triggered the episode
    - Content node: what happened (the agent's output)
    - Outcome node: result (success/failure/corrected)
    """

    def __init__(
        self,
        graph: GraphStore,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        self._graph = graph
        self._embedder = embedding_provider
        self._episodes: dict[str, Episode] = {}

    async def record(self, episode: Episode) -> str:
        cue_node = MemoryNode(
            content=episode.cue,
            node_type=MemoryType.EPISODIC,
            id=f"{episode.id}_cue",
            embedding=await self._embedder.embed(episode.cue),
            metadata={
                "episode_id": episode.id,
                "role": "cue",
                "task_type": episode.task_type or "",
                "tags": episode.tags,
                "input_data": episode.input_data,
            },
        )
        content_parts = [json.dumps(episode.content, default=str)]
        if episode.tool_calls:
            tool_lines = []
            for i, tc in enumerate(episode.tool_calls):
                tool_lines.append(
                    f"[{i}] {tc.server}/{tc.tool_name} "
                    f"query={tc.query!r} success={tc.success} "
                    f"params={json.dumps(tc.parameters, default=str)} "
                    f"result={tc.result_summary[:200]}"
                )
            content_parts.append("Tool calls:\n" + "\n".join(tool_lines))
        content_str = "\n\n".join(content_parts)
        content_node = MemoryNode(
            content=content_str,
            node_type=MemoryType.EPISODIC,
            id=f"{episode.id}_content",
            metadata={
                "episode_id": episode.id,
                "role": "content",
                "tool_names": [tc.tool_name for tc in episode.tool_calls],
                "episode_content": episode.content,
                "tool_calls": [
                    {
                        "query": tc.query,
                        "server": tc.server,
                        "tool_name": tc.tool_name,
                        "parameters": tc.parameters,
                        "result_summary": tc.result_summary,
                        "success": tc.success,
                    }
                    for tc in episode.tool_calls
                ],
            },
        )
        outcome_str = episode.outcome.value
        if episode.correction:
            outcome_str += f" | correction: {episode.correction}"
        outcome_node = MemoryNode(
            content=outcome_str,
            node_type=MemoryType.EPISODIC,
            id=f"{episode.id}_outcome",
            metadata={
                "episode_id": episode.id,
                "role": "outcome",
                "outcome": episode.outcome.value,
                "correction": episode.correction,
            },
        )
        edges = [
            MemoryEdge(
                source_id=cue_node.id,
                target_id=content_node.id,
                relation_type="produced",
            ),
            MemoryEdge(
                source_id=content_node.id,
                target_id=outcome_node.id,
                relation_type="resulted_in",
            ),
        ]
        if episode.outcome == EpisodeOutcome.CORRECTED and episode.correction:
            edges.append(
                MemoryEdge(
                    source_id=outcome_node.id,
                    target_id=cue_node.id,
                    relation_type="corrected_to",
                    metadata={"correction": episode.correction},
                )
            )

        # One atomic write — a mid-write failure must not leave a sheared episode.
        await self._graph.write_batch(
            [cue_node, content_node, outcome_node], edges
        )
        self._episodes[episode.id] = episode
        return episode.id

    async def recall(
        self,
        cue: str,
        task_type: str | None = None,
        outcome_filter: EpisodeOutcome | None = None,
        max_episodes: int = 3,
        capacity: CapacityHints | None = None,
    ) -> list[Episode]:
        if capacity and capacity.recommended_chunk_tokens < 2000:
            max_episodes = min(max_episodes, 2)

        cue_nodes = await self._find_matching_cues(cue, task_type=task_type)

        episodes: list[tuple[float, Episode]] = []
        for sim, cue_node in cue_nodes:
            episode = await self._reconstruct_episode(cue_node)
            if episode is None:
                continue
            if outcome_filter and episode.outcome != outcome_filter:
                continue

            score = sim
            if episode.outcome == EpisodeOutcome.FAILURE:
                score *= 1.5
            elif episode.outcome == EpisodeOutcome.CORRECTED:
                score *= 1.3

            episodes.append((score, episode))

        episodes.sort(key=lambda x: x[0], reverse=True)
        results = [ep for _, ep in episodes[:max_episodes]]

        for ep in results:
            await self._graph.update_utility(f"{ep.id}_cue")

        return results

    async def _reconstruct_episode(self, cue_node: MemoryNode) -> Episode | None:
        """Reconstruct an Episode from its graph subgraph (cue→content→outcome)."""
        episode_id = cue_node.metadata.get("episode_id")
        if episode_id is None:
            return None

        # Session cache hit
        if episode_id in self._episodes:
            return self._episodes[episode_id]

        # Traverse cue→produced→content→resulted_in→outcome
        traversed = await self._graph.traverse(
            start=cue_node.id,
            edge_types=["produced", "resulted_in"],
            max_depth=2,
            max_nodes=10,
        )

        content_node = None
        outcome_node = None
        for node in traversed:
            role = node.metadata.get("role")
            if role == "content" and node.metadata.get("episode_id") == episode_id:
                content_node = node
            elif role == "outcome" and node.metadata.get("episode_id") == episode_id:
                outcome_node = node

        if content_node is None or outcome_node is None:
            return None

        # Reconstruct tool calls from structured metadata
        tool_calls = [
            ToolCall(**tc_data)
            for tc_data in content_node.metadata.get("tool_calls", [])
        ]

        episode = Episode(
            id=episode_id,
            cue=cue_node.content,
            content=content_node.metadata.get("episode_content", {}),
            outcome=EpisodeOutcome(outcome_node.metadata["outcome"]),
            correction=outcome_node.metadata.get("correction"),
            task_type=cue_node.metadata.get("task_type") or None,
            timestamp=cue_node.created_at,
            input_data=cue_node.metadata.get("input_data"),
            tags=cue_node.metadata.get("tags", []),
            tool_calls=tool_calls,
        )

        # Cache for this session
        self._episodes[episode_id] = episode
        return episode

    async def find_patterns(
        self,
        task_type: str,
        min_occurrences: int = 3,
    ) -> list[EpisodicPattern]:
        episodes = await self._episodes_for_task_type(task_type)

        by_outcome: dict[EpisodeOutcome, list[Episode]] = {}
        for episode in episodes:
            by_outcome.setdefault(episode.outcome, []).append(episode)

        patterns: list[EpisodicPattern] = []
        for outcome, eps in by_outcome.items():
            if len(eps) >= min_occurrences:
                patterns.append(
                    EpisodicPattern(
                        task_type=task_type,
                        pattern_description=f"{len(eps)} episodes with outcome={outcome.value}",
                        occurrences=len(eps),
                        episode_ids=[e.id for e in eps],
                        outcome=outcome,
                    )
                )

        # Tool-specific failure patterns: group by (tool_name, success=False)
        tool_failures: dict[str, list[Episode]] = {}
        for episode in episodes:
            for tc in episode.tool_calls:
                if not tc.success:
                    tool_failures.setdefault(tc.tool_name, []).append(episode)

        for tool_name, eps in tool_failures.items():
            # Deduplicate episodes (one episode may have multiple failed calls to same tool)
            unique_eps = list({e.id: e for e in eps}.values())
            if len(unique_eps) >= min_occurrences:
                patterns.append(
                    EpisodicPattern(
                        task_type=task_type,
                        pattern_description=f"Tool '{tool_name}' failed in {len(unique_eps)} episodes",
                        occurrences=len(unique_eps),
                        episode_ids=[e.id for e in unique_eps],
                        outcome=EpisodeOutcome.FAILURE,
                    )
                )

        return patterns

    def generate_warnings(self, episodes: list[Episode]) -> list[str]:
        warnings = []
        for ep in episodes:
            if ep.outcome == EpisodeOutcome.FAILURE:
                warnings.append(
                    f"Previous failure on similar task ({ep.task_type}): {ep.cue}"
                )
            elif ep.outcome == EpisodeOutcome.CORRECTED:
                warnings.append(
                    f"Previous correction on similar task ({ep.task_type}): "
                    f"{ep.correction}"
                )
        return warnings

    async def _find_matching_cues(
        self,
        query: str,
        task_type: str | None = None,
        limit: int = 10,
    ) -> list[tuple[float, MemoryNode]]:
        query_embedding = await self._embedder.embed(query)

        similar = await self._graph.similarity_search(
            query_embedding, node_type=MemoryType.EPISODIC, limit=limit * 5
        )

        candidates: list[tuple[float, MemoryNode]] = []
        for node in similar:
            if node.metadata.get("role") != "cue":
                continue
            if task_type and node.metadata.get("task_type") != task_type:
                continue
            sim = cosine_similarity(query_embedding, node.embedding) if node.embedding else 0.0
            candidates.append((sim, node))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:limit]

    async def episode_count(self) -> int:
        cue_nodes = await self._graph.query(
            node_type=MemoryType.EPISODIC,
            filters={"role": "cue"},
        )
        return len(cue_nodes)

    async def all_episodes(self) -> list[Episode]:
        """Reconstruct all episodes from the graph."""
        cue_nodes = await self._graph.query(
            node_type=MemoryType.EPISODIC,
            filters={"role": "cue"},
        )
        episodes = []
        for cue_node in cue_nodes:
            episode = await self._reconstruct_episode(cue_node)
            if episode is not None:
                episodes.append(episode)
        return episodes

    async def _episodes_for_task_type(self, task_type: str) -> list[Episode]:
        """Reconstruct all episodes for a given task type from the graph."""
        cue_nodes = await self._graph.query(
            node_type=MemoryType.EPISODIC,
            filters={"role": "cue", "task_type": task_type},
        )
        episodes = []
        for cue_node in cue_nodes:
            episode = await self._reconstruct_episode(cue_node)
            if episode is not None:
                episodes.append(episode)
        return episodes
