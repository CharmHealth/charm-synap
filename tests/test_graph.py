"""Tests for MemoryGraph — the core data structure."""

from synap.graph import MemoryGraph
from synap.types import MemoryEdge, MemoryNode, MemoryType


async def test_add_and_get_node(graph: MemoryGraph):
    node = MemoryNode(content="test fact", node_type=MemoryType.SEMANTIC)
    await graph.add_node(node)
    assert await graph.get_node(node.id) is node
    assert await graph.node_count() == 1
    assert await graph.node_count(MemoryType.SEMANTIC) == 1
    assert await graph.node_count(MemoryType.EPISODIC) == 0


async def test_add_edge_validates_nodes(graph: MemoryGraph):
    n1 = MemoryNode(content="a", node_type=MemoryType.SEMANTIC)
    n2 = MemoryNode(content="b", node_type=MemoryType.SEMANTIC)
    await graph.add_node(n1)
    await graph.add_node(n2)

    edge = MemoryEdge(source_id=n1.id, target_id=n2.id, relation_type="related_to")
    await graph.add_edge(edge)
    assert await graph.edge_count() == 1

    # Edge to nonexistent node should raise
    bad_edge = MemoryEdge(source_id=n1.id, target_id="nonexistent", relation_type="x")
    try:
        await graph.add_edge(bad_edge)
        raise AssertionError("Should have raised KeyError")
    except KeyError:
        pass


async def test_neighbors(graph: MemoryGraph):
    n1 = MemoryNode(content="a", node_type=MemoryType.SEMANTIC)
    n2 = MemoryNode(content="b", node_type=MemoryType.SEMANTIC)
    n3 = MemoryNode(content="c", node_type=MemoryType.SEMANTIC)
    for n in [n1, n2, n3]:
        await graph.add_node(n)

    await graph.add_edge(MemoryEdge(source_id=n1.id, target_id=n2.id, relation_type="causes"))
    await graph.add_edge(MemoryEdge(source_id=n1.id, target_id=n3.id, relation_type="related_to"))

    # Outgoing neighbors of n1
    neighbors = await graph.neighbors(n1.id)
    assert len(neighbors) == 2

    # Filtered by edge type
    causal = await graph.neighbors(n1.id, edge_type="causes")
    assert len(causal) == 1
    assert causal[0].id == n2.id


async def test_traverse_bfs(graph: MemoryGraph):
    # Build a chain: a → b → c → d
    nodes = [MemoryNode(content=f"node_{i}", node_type=MemoryType.SEMANTIC) for i in range(4)]
    for n in nodes:
        await graph.add_node(n)
    for i in range(3):
        await graph.add_edge(
            MemoryEdge(source_id=nodes[i].id, target_id=nodes[i + 1].id, relation_type="next")
        )

    # Depth 1 from node 0 → only node 1
    result = await graph.traverse(nodes[0].id, max_depth=1)
    assert len(result) == 1
    assert result[0].id == nodes[1].id

    # Depth 3 from node 0 → nodes 1, 2, 3
    result = await graph.traverse(nodes[0].id, max_depth=3)
    assert len(result) == 3


async def test_traverse_max_nodes(graph: MemoryGraph):
    # Hub: center → 10 spokes
    center = MemoryNode(content="center", node_type=MemoryType.SEMANTIC)
    await graph.add_node(center)
    for i in range(10):
        spoke = MemoryNode(content=f"spoke_{i}", node_type=MemoryType.SEMANTIC)
        await graph.add_node(spoke)
        await graph.add_edge(
            MemoryEdge(source_id=center.id, target_id=spoke.id, relation_type="has")
        )

    result = await graph.traverse(center.id, max_nodes=3)
    assert len(result) == 3


async def test_remove_node_cleans_edges(graph: MemoryGraph):
    n1 = MemoryNode(content="a", node_type=MemoryType.SEMANTIC)
    n2 = MemoryNode(content="b", node_type=MemoryType.SEMANTIC)
    await graph.add_node(n1)
    await graph.add_node(n2)
    await graph.add_edge(MemoryEdge(source_id=n1.id, target_id=n2.id, relation_type="x"))

    await graph.remove_node(n2.id)
    assert await graph.node_count() == 1
    assert await graph.edge_count() == 0
    assert await graph.neighbors(n1.id) == []


async def test_utility_and_eviction(graph: MemoryGraph):
    n1 = MemoryNode(content="active", node_type=MemoryType.SEMANTIC)
    n2 = MemoryNode(content="stale", node_type=MemoryType.SEMANTIC)
    n2.utility_score = 0.05  # Below default threshold

    await graph.add_node(n1)
    await graph.add_node(n2)

    evicted = await graph.evict(threshold=0.1)
    assert n2.id in evicted
    assert await graph.node_count() == 1
    assert await graph.get_node(n1.id) is not None


async def test_edges_between(graph: MemoryGraph):
    n1 = MemoryNode(content="a", node_type=MemoryType.SEMANTIC)
    n2 = MemoryNode(content="b", node_type=MemoryType.SEMANTIC)
    await graph.add_node(n1)
    await graph.add_node(n2)

    await graph.add_edge(MemoryEdge(source_id=n1.id, target_id=n2.id, relation_type="causes"))
    await graph.add_edge(MemoryEdge(source_id=n1.id, target_id=n2.id, relation_type="related_to"))

    all_edges = await graph.edges_between(n1.id, n2.id)
    assert len(all_edges) == 2

    causal = await graph.edges_between(n1.id, n2.id, relation_type="causes")
    assert len(causal) == 1


async def test_has_incoming_edge(graph: MemoryGraph):
    n1 = MemoryNode(content="v2", node_type=MemoryType.PROCEDURAL)
    n2 = MemoryNode(content="v1", node_type=MemoryType.PROCEDURAL)
    await graph.add_node(n1)
    await graph.add_node(n2)

    assert not await graph.has_incoming_edge(n2.id, "supersedes")

    await graph.add_edge(MemoryEdge(source_id=n1.id, target_id=n2.id, relation_type="supersedes"))
    assert await graph.has_incoming_edge(n2.id, "supersedes")
    assert not await graph.has_incoming_edge(n1.id, "supersedes")


async def test_cross_partition_edges(graph: MemoryGraph):
    """Edges can connect nodes across memory type partitions."""
    semantic = MemoryNode(content="fact", node_type=MemoryType.SEMANTIC)
    episodic = MemoryNode(content="experience", node_type=MemoryType.EPISODIC)
    await graph.add_node(semantic)
    await graph.add_node(episodic)

    edge = MemoryEdge(
        source_id=semantic.id, target_id=episodic.id, relation_type="derived_from"
    )
    await graph.add_edge(edge)

    neighbors = await graph.neighbors(semantic.id)
    assert len(neighbors) == 1
    assert neighbors[0].node_type == MemoryType.EPISODIC
