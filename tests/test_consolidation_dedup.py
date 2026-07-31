"""Consolidating the same pattern twice yields one fact (CH-728 item 6).

A consolidated fact's identity is its pattern, so re-consolidating the same
pattern must upsert the same node, not spray near-duplicate facts with fresh
random UUIDs. Identity comes from a stable pattern key (task_type + outcome /
tool), never the episode count, so it survives the pattern growing.
"""

from __future__ import annotations

from synap.consolidation import ConsolidationEngine
from synap.episodic import EpisodicMemory
from synap.graph import MemoryGraph
from synap.procedural import ProceduralMemory
from synap.semantic import SemanticMemory
from synap.types import (
    ConsolidationEvent,
    ConsolidationTrigger,
    MemoryNode,
    MemoryType,
)

from tests.conftest import FakeEmbedder, FakeLLM


def _episode_content(nid: str) -> MemoryNode:
    return MemoryNode(
        id=f"{nid}_content",
        node_type=MemoryType.EPISODIC,
        content=f"experience {nid}: billing check succeeded",
        embedding=[0.1, 0.2, 0.3],
        metadata={"episode_id": nid},
    )


def _engine(graph: MemoryGraph):
    embedder, llm = FakeEmbedder(), FakeLLM()
    return ConsolidationEngine(
        graph=graph,
        domain=SemanticMemory(graph, embedder, llm),
        procedural=ProceduralMemory(graph, embedder),
        episodic=EpisodicMemory(graph, embedder),
        llm_provider=llm,
    )


def _success_event(candidates) -> ConsolidationEvent:
    return ConsolidationEvent(
        source_type=MemoryType.EPISODIC,
        target_type=MemoryType.SEMANTIC,
        candidates=candidates,
        trigger=ConsolidationTrigger.PERIODIC,
        confidence=0.8,
        metadata={"task_type": "billing", "pattern_key": "outcome:success"},
    )


async def test_same_pattern_consolidated_twice_yields_one_fact():
    graph = MemoryGraph()
    engine = _engine(graph)
    candidates = [_episode_content(f"e{i}") for i in range(3)]
    for c in candidates:
        await graph.add_node(c)

    await engine.process(_success_event(candidates))
    await engine.process(_success_event(candidates))  # same pattern again

    count = await graph.node_count(MemoryType.SEMANTIC)
    assert count == 1, f"expected one consolidated fact, got {count}"


async def test_growing_pattern_stays_one_fact():
    """The pattern gaining an episode must not create a second fact — identity
    is the stable key, not the episode set or count."""
    graph = MemoryGraph()
    engine = _engine(graph)
    first = [_episode_content(f"e{i}") for i in range(3)]
    for c in first:
        await graph.add_node(c)
    await engine.process(_success_event(first))

    grown = first + [_episode_content("e3")]
    await graph.add_node(grown[-1])
    await engine.process(_success_event(grown))

    count = await graph.node_count(MemoryType.SEMANTIC)
    assert count == 1, f"growing pattern duplicated the fact: {count} facts"
