"""Tests for PersistentGraph — runs subsystem tests through a Kùzu backend."""

from pathlib import Path

import pytest

from synap.backends.kuzu import KuzuBackend
from synap.persistent_graph import PersistentGraph
from synap.types import MemoryEdge, MemoryNode, MemoryType


@pytest.fixture
def pgraph(tmp_path: Path) -> PersistentGraph:
    backend = KuzuBackend(tmp_path / "test_graph", embedding_dim=8)
    pg = PersistentGraph(backend=backend)
    yield pg
    backend.close()


def _node(
    id: str = "n1",
    content: str = "test fact",
    node_type: MemoryType = MemoryType.SEMANTIC,
    embedding: list[float] | None = None,
) -> MemoryNode:
    return MemoryNode(
        id=id,
        content=content,
        node_type=node_type,
        embedding=embedding or [0.1] * 8,
    )


def _edge(
    id: str = "e1",
    source_id: str = "n1",
    target_id: str = "n2",
    relation_type: str = "related_to",
) -> MemoryEdge:
    return MemoryEdge(
        id=id,
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
    )


async def test_add_and_get_node(pgraph: PersistentGraph):
    node = _node("n1", content="hello world")
    await pgraph.add_node(node)

    loaded = await pgraph.get_node("n1")
    assert loaded is not None
    assert loaded.content == "hello world"
    assert loaded.node_type == MemoryType.SEMANTIC


async def test_get_nonexistent_node(pgraph: PersistentGraph):
    assert await pgraph.get_node("nope") is None


async def test_add_edge_validates_nodes(pgraph: PersistentGraph):
    with pytest.raises(KeyError):
        await pgraph.add_edge(_edge("e1", "missing_a", "missing_b"))


async def test_add_and_traverse(pgraph: PersistentGraph):
    await pgraph.add_node(_node("n1"))
    await pgraph.add_node(_node("n2"))
    await pgraph.add_node(_node("n3"))
    await pgraph.add_edge(_edge("e1", "n1", "n2"))
    await pgraph.add_edge(_edge("e2", "n2", "n3"))

    result = await pgraph.traverse("n1", max_depth=1)
    ids = {r.id for r in result}
    assert "n2" in ids
    assert "n3" not in ids

    result = await pgraph.traverse("n1", max_depth=2)
    ids = {r.id for r in result}
    assert "n2" in ids
    assert "n3" in ids


async def test_query_by_type(pgraph: PersistentGraph):
    await pgraph.add_node(_node("n1", node_type=MemoryType.SEMANTIC))
    await pgraph.add_node(_node("n2", node_type=MemoryType.EPISODIC))
    await pgraph.add_node(_node("n3", node_type=MemoryType.SEMANTIC))

    semantic = await pgraph.query(node_type=MemoryType.SEMANTIC)
    assert len(semantic) == 2

    episodic = await pgraph.query(node_type=MemoryType.EPISODIC)
    assert len(episodic) == 1


async def test_query_with_filters(pgraph: PersistentGraph):
    n1 = _node("n1")
    n1.metadata = {"tag": "important"}
    await pgraph.add_node(n1)

    n2 = _node("n2")
    n2.metadata = {"tag": "trivial"}
    await pgraph.add_node(n2)

    results = await pgraph.query(filters={"tag": "important"})
    assert len(results) == 1
    assert results[0].id == "n1"


async def test_remove_node_cleans_edges(pgraph: PersistentGraph):
    await pgraph.add_node(_node("n1"))
    await pgraph.add_node(_node("n2"))
    await pgraph.add_edge(_edge("e1", "n1", "n2"))

    await pgraph.remove_node("n1")
    assert await pgraph.get_node("n1") is None

    # n2's edges should be gone too
    result = await pgraph.traverse("n2", max_depth=1)
    assert len(result) == 0


async def test_edges_between(pgraph: PersistentGraph):
    await pgraph.add_node(_node("n1"))
    await pgraph.add_node(_node("n2"))
    await pgraph.add_edge(_edge("e1", "n1", "n2", "causes"))
    await pgraph.add_edge(_edge("e2", "n1", "n2", "related_to"))

    edges = await pgraph.edges_between("n1", "n2")
    assert len(edges) == 2

    causal = await pgraph.edges_between("n1", "n2", relation_type="causes")
    assert len(causal) == 1
    assert causal[0].relation_type == "causes"


async def test_has_incoming_edge(pgraph: PersistentGraph):
    await pgraph.add_node(_node("n1"))
    await pgraph.add_node(_node("n2"))
    await pgraph.add_edge(_edge("e1", "n1", "n2", "supersedes"))

    assert await pgraph.has_incoming_edge("n2", "supersedes") is True
    assert await pgraph.has_incoming_edge("n1", "supersedes") is False


async def test_node_count(pgraph: PersistentGraph):
    await pgraph.add_node(_node("n1", node_type=MemoryType.SEMANTIC))
    await pgraph.add_node(_node("n2", node_type=MemoryType.EPISODIC))

    assert await pgraph.node_count() == 2
    assert await pgraph.node_count(MemoryType.SEMANTIC) == 1


async def test_edge_count(pgraph: PersistentGraph):
    await pgraph.add_node(_node("n1"))
    await pgraph.add_node(_node("n2"))
    await pgraph.add_edge(_edge("e1", "n1", "n2", "causes"))
    await pgraph.add_edge(_edge("e2", "n1", "n2", "related_to"))

    assert await pgraph.edge_count() == 2
    assert await pgraph.edge_count("causes") == 1


async def test_update_utility(pgraph: PersistentGraph):
    await pgraph.add_node(_node("n1"))
    original = await pgraph.get_node("n1")
    original_count = original.access_count

    await pgraph.update_utility("n1")

    updated = await pgraph.get_node("n1")
    assert updated.access_count == original_count + 1


async def test_decay_all_server_side(pgraph: PersistentGraph):
    """decay_all uses server-side Cypher and produces correct scores."""
    from datetime import timedelta

    n1 = _node("n1")
    n1.last_accessed = n1.last_accessed - timedelta(hours=10)
    n1.access_count = 5
    await pgraph.add_node(n1)

    n2 = _node("n2")
    n2.last_accessed = n2.last_accessed - timedelta(hours=100)
    n2.access_count = 25
    await pgraph.add_node(n2)

    await pgraph.decay_all()

    r1 = await pgraph.get_node("n1")
    r2 = await pgraph.get_node("n2")

    # Verify via Python formula
    from synap.persistent_graph import compute_decay_score

    expected_1 = compute_decay_score(10.0, 5, pgraph._utility_decay_rate)
    expected_2 = compute_decay_score(100.0, 25, pgraph._utility_decay_rate)

    assert abs(r1.utility_score - expected_1) < 1e-6, (
        f"n1: got {r1.utility_score}, expected {expected_1}"
    )
    assert abs(r2.utility_score - expected_2) < 1e-6, (
        f"n2: got {r2.utility_score}, expected {expected_2}"
    )


def test_decay_formula_sync():
    """Python compute_decay_score matches expected values across a spread of inputs."""
    from synap.persistent_graph import compute_decay_score

    cases = [
        # (hours, access_count, rate)
        (0.0, 0, 0.01),      # near-zero hours (clamped), zero access
        (1.0, 10, 0.01),     # 1 hour, moderate access
        (24.0, 20, 0.01),    # 1 day, saturated frequency
        (168.0, 5, 0.01),    # 1 week
        (720.0, 50, 0.01),   # 1 month, over-saturated
    ]
    import math
    for hours, count, rate in cases:
        result = compute_decay_score(hours, count, rate)
        h = max(1.0 / 3600, hours)
        # recency gates frequency: r * (1 + f), not r + f
        expected = math.pow(1 - rate, h) * (1 + min(1.0, count / 20))
        assert abs(result - expected) < 1e-12, (
            f"hours={hours}, count={count}: got {result}, expected {expected}"
        )


async def test_evict_server_side(pgraph: PersistentGraph):
    """evict uses server-side query and removes low-score nodes."""
    n1 = _node("n1")
    n1.utility_score = 0.05  # below threshold
    await pgraph.add_node(n1)

    n2 = _node("n2")
    n2.utility_score = 0.5  # above threshold
    await pgraph.add_node(n2)

    evicted = await pgraph.evict(threshold=0.1)
    assert "n1" in evicted
    assert "n2" not in evicted
    assert await pgraph.get_node("n1") is None
    assert await pgraph.get_node("n2") is not None


async def test_facade_with_kuzu_backend(tmp_path: Path):
    """Full integration: CognitiveMemory with Kùzu backend."""
    from tests.conftest import FakeEmbedder, FakeLLM
    from synap.facade import CognitiveMemory
    from synap.semantic import SemanticMemory
    from synap.types import EpisodeOutcome, Procedure

    backend = KuzuBackend(tmp_path / "facade_test", embedding_dim=8)
    embedder = FakeEmbedder()
    pg = PersistentGraph(backend=backend)
    domain = SemanticMemory(graph=pg, embedding_provider=embedder)

    cm = CognitiveMemory(
        domain=domain,
        embedding_provider=embedder,
        llm_provider=FakeLLM(),
        graph=pg,
    )

    proc = Procedure(
        task_type="diagnosis",
        description="Medical diagnosis procedure",
        schema={
            "symptoms": {"type": "string"},
            "reasoning": {"type": "string"},
            "diagnosis": {"type": "string"},
        },
        field_ordering=["symptoms", "reasoning", "diagnosis"],
    )
    await cm.procedural.register(proc)

    await domain.store("Fever and cough suggest respiratory infection")

    ctx = await cm.prepare_call("diagnosis for patient with fever")
    assert ctx.procedure is not None
    assert ctx.output_schema is not None

    episode_id = await cm.record_outcome(
        task_description="diagnosis for patient with fever",
        input_data={"symptoms": "fever, cough"},
        output={"diagnosis": "respiratory infection"},
        outcome=EpisodeOutcome.SUCCESS,
        task_type="diagnosis",
    )
    assert episode_id is not None

    stats = await cm.stats()
    assert stats.semantic_nodes >= 1
    assert stats.procedural_nodes >= 1
    assert stats.total_episodes == 1

    backend.close()
