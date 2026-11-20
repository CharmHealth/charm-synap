"""Tenant isolation: a read scoped to one tenant never returns another's memory.

Every test runs against the in-memory graph and a PersistentGraph over each
backend (Kuzu, SQLite, and Postgres when SYNAP_TEST_POSTGRES_DSN is set), since
the filter lives in each backend's query and a gap in one would not show in
the others.
"""

from __future__ import annotations

import os
import uuid

import pytest

from synap.backends.kuzu import KuzuBackend
from synap.backends.sqlite import SQLiteBackend
from synap.consolidation import ConsolidationResult
from synap.facade import CognitiveMemory
from synap.graph import MemoryGraph
from synap.persistent_graph import PersistentGraph
from synap.semantic import SemanticMemory
from synap.tenancy import TenantMismatchError
from synap.types import EpisodeOutcome, MemoryEdge, MemoryNode, MemoryType, Procedure
from tests.conftest import FakeEmbedder, FakeLLM

try:
    import asyncpg

    from synap.backends.postgres import PostgresBackend
except ImportError:  # pragma: no cover
    asyncpg = None

DSN = os.environ.get("SYNAP_TEST_POSTGRES_DSN")
DIM = 8  # FakeEmbedder's dimension


@pytest.fixture(params=["memory", "kuzu", "sqlite", "postgres"])
async def root(request, tmp_path):
    """An unscoped graph; tests scope it themselves."""
    name = request.param
    if name == "memory":
        yield MemoryGraph()
    elif name == "kuzu":
        g = PersistentGraph(KuzuBackend(tmp_path / "graph", embedding_dim=DIM))
        try:
            yield g
        finally:
            g.close()
    elif name == "sqlite":
        g = PersistentGraph(SQLiteBackend(tmp_path / "graph.db"))
        try:
            yield g
        finally:
            g.close()
    else:
        if asyncpg is None or not DSN:
            pytest.skip("set SYNAP_TEST_POSTGRES_DSN (with asyncpg) for Postgres")
        prefix = f"t_{uuid.uuid4().hex[:12]}_"
        pool = await asyncpg.create_pool(DSN, min_size=1, max_size=5)
        backend = PostgresBackend(pool, embedding_dim=DIM, table_prefix=prefix)
        await backend.init()
        try:
            yield PersistentGraph(backend)
        finally:
            async with pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {prefix}edges CASCADE")
                await conn.execute(f"DROP TABLE IF EXISTS {prefix}nodes CASCADE")
            await pool.close()


def _node(nid: str, embedding: list[float], **meta) -> MemoryNode:
    return MemoryNode(
        id=nid,
        content=f"content of {nid}",
        node_type=MemoryType.SEMANTIC,
        embedding=embedding,
        metadata={"kind": "fact", **meta},
    )


NEAR = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
FAR = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


async def _populate(root):
    """Tenant A: two nodes far from NEAR, linked. Tenant B: five nodes at NEAR,
    linked to each other, so on any unscoped ranking B's nodes would come first."""
    a, b = root.scoped("A"), root.scoped("B")
    await a.add_node(_node("a1", FAR))
    await a.add_node(_node("a2", FAR))
    await a.add_edge(MemoryEdge(source_id="a1", target_id="a2", relation_type="rel"))
    for i in range(5):
        await b.add_node(_node(f"b{i}", NEAR))
    for i in range(4):
        await b.add_edge(
            MemoryEdge(source_id=f"b{i}", target_id=f"b{i + 1}", relation_type="rel")
        )
    await root.add_node(_node("u1", NEAR))  # untenanted
    return a, b


# --- Reads ------------------------------------------------------------------


async def test_similarity_ranks_within_tenant_only(root):
    a, _ = await _populate(root)
    # B's five nodes are all closer than A's. A post-filter over a global
    # top-2 would return nothing; the filter must run before the limit.
    hits = await a.similarity_search(NEAR, limit=2)
    assert sorted(n.id for n in hits) == ["a1", "a2"]
    assert all(n.tenant == "A" for n in hits)


async def test_query_and_count_within_tenant_only(root):
    a, b = await _populate(root)
    assert sorted(n.id for n in await a.query(filters={"kind": "fact"}, limit=2)) == [
        "a1",
        "a2",
    ]
    assert await a.node_count() == 2
    assert await b.node_count(MemoryType.SEMANTIC) == 5
    assert await a.edge_count() == 1
    assert await b.edge_count("rel") == 4


async def test_get_node_by_id_hides_other_tenants(root):
    a, b = await _populate(root)
    assert await a.get_node("b0") is None
    assert await a.get_node("u1") is None
    assert (await b.get_node("b0")).tenant == "B"


async def test_unscoped_graph_sees_only_untenanted(root):
    await _populate(root)
    assert [n.id for n in await root.query()] == ["u1"]
    assert [n.id for n in await root.similarity_search(NEAR, limit=10)] == ["u1"]
    assert await root.get_node("a1") is None
    assert await root.node_count() == 1


async def test_graph_walks_stay_in_tenant(root):
    a, _ = await _populate(root)
    assert await a.traverse("b0") == []
    assert await a.neighbors("b0", direction="both") == []
    assert await a.edges_between("b0", "b1") == []
    assert await a.has_incoming_edge("b1", "rel") is False
    assert [n.id for n in await a.traverse("a1")] == ["a2"]


async def test_mutations_cannot_reach_other_tenants(root):
    a, b = await _populate(root)
    await a.remove_node("b0")
    await a.update_utility("b0")
    assert (await b.get_node("b0")).access_count == 0
    await a.evict(threshold=2.0)  # would evict everything visible
    assert await b.node_count() == 5
    assert await root.node_count() == 1


async def test_tenants_lists_owners(root):
    await _populate(root)
    assert await root.tenants() == ["A", "B"]


# --- Writes -----------------------------------------------------------------


async def test_write_cannot_overwrite_another_tenants_node(root):
    a, b = await _populate(root)
    with pytest.raises(TenantMismatchError):
        await a.add_node(_node("b0", FAR))
    with pytest.raises(TenantMismatchError):
        await a.write_batch([_node("b1", FAR)], [])
    with pytest.raises(TenantMismatchError):
        await root.add_node(_node("a1", FAR))
    assert (await b.get_node("b0")).embedding == NEAR


async def test_write_refuses_node_stamped_for_another_tenant(root):
    a, _ = await _populate(root)
    node = _node("x", FAR)
    node.tenant = "B"
    with pytest.raises(TenantMismatchError):
        await a.add_node(node)


async def test_write_batch_does_not_stamp_nodes_on_a_failed_edge_validation(root):
    """A write_batch that never lands must not leave the caller's node object
    looking written — it's still usable, and still unstamped, for a retry."""
    a, _ = await _populate(root)
    node = _node("a3", FAR)
    assert node.tenant is None
    with pytest.raises(KeyError):
        await a.write_batch(
            [node], [MemoryEdge(source_id="a3", target_id="b0", relation_type="rel")]
        )
    assert node.tenant is None
    assert await a.get_node("a3") is None


async def test_edge_cannot_join_two_tenants(root):
    a, _ = await _populate(root)
    with pytest.raises(KeyError):
        await a.add_edge(MemoryEdge(source_id="a1", target_id="b0", relation_type="rel"))
    with pytest.raises(KeyError):
        await a.write_batch(
            [_node("a3", FAR)],
            [MemoryEdge(source_id="a3", target_id="b0", relation_type="rel")],
        )


async def test_scoping_rules(root):
    a = root.scoped("A")
    assert a.tenant == "A" and root.tenant is None
    with pytest.raises(ValueError):
        a.scoped("B")
    for bad in ("", "   ", None):
        with pytest.raises(ValueError):
            root.scoped(bad)


# --- Facade -----------------------------------------------------------------


def _memory(graph) -> CognitiveMemory:
    embedder, llm = FakeEmbedder(), FakeLLM()
    return CognitiveMemory(
        domain=SemanticMemory(graph, embedder, llm),
        embedding_provider=embedder,
        llm_provider=llm,
        graph=graph,
    )


async def _record(memory: CognitiveMemory, n: int, outcome=EpisodeOutcome.SUCCESS):
    ids = []
    for i in range(n):
        ids.append(
            await memory.record_outcome(
                task_description=f"verify billing code {i}",
                input_data={"i": i},
                output={"ok": True},
                outcome=outcome,
                task_type="billing",
            )
        )
    return ids


async def test_facade_episodes_procedures_and_knowledge_isolated(root):
    memory = _memory(root)
    a, b = memory.scoped("A"), memory.scoped("B")
    assert memory.scoped("A") is a  # cached

    await _record(a, 3)
    await a.domain.absorb(insights=["Payer X needs form 12"], source_episodes=[])
    await a.procedural.register(
        Procedure(
            task_type="billing",
            description="billing procedure",
            schema={},
            field_ordering=[],
        )
    )

    assert len(await a.episodic.recall("verify billing code", max_episodes=10)) == 3
    assert await b.episodic.recall("verify billing code", max_episodes=10) == []
    assert await b.procedural.match("billing", task_type="billing") is None
    assert (await a.procedural.match("billing", task_type="billing")) is not None
    assert await b.domain.retrieve("Payer X form") == []
    prepared = await b.prepare_call("verify billing code 1", task_type="billing")
    assert prepared.procedure is None
    assert prepared.relevant_episodes == []
    assert prepared.domain_context == []
    assert (await memory.stats()).episodic_nodes == 0


async def test_consolidation_runs_per_tenant(root):
    memory = _memory(root)
    a, b = memory.scoped("A"), memory.scoped("B")
    a_ids = await _record(a, 3)
    b_ids = await _record(b, 3)

    results = await memory.consolidate_tenants()
    assert set(results) == {"A", "B"}

    for scoped, own, other in ((a, a_ids, b_ids), (b, b_ids, a_ids)):
        facts = [
            n
            for n in await scoped.graph.query(node_type=MemoryType.SEMANTIC)
            if n.id.startswith("consolidated:")
        ]
        assert facts, "each tenant gets its own consolidated fact"
        for fact in facts:
            sources = {s.removesuffix("_content") for s in fact.metadata["consolidated_from"]}
            assert sources <= set(own)
            assert not sources & set(other)

    # Same pattern in both tenants: distinct facts, not one shared id.
    a_fact = {n.id for n in await a.graph.query(node_type=MemoryType.SEMANTIC)}
    b_fact = {n.id for n in await b.graph.query(node_type=MemoryType.SEMANTIC)}
    assert not a_fact & b_fact


async def test_consolidate_tenants_continues_past_a_failing_tenant(root):
    """One tenant's consolidation raising must not lose every other tenant's
    results, and the failure must show up in the dict rather than vanish."""
    memory = _memory(root)
    a, b = memory.scoped("A"), memory.scoped("B")
    await _record(a, 3)
    await _record(b, 3)

    async def boom() -> list[ConsolidationResult]:
        raise RuntimeError("boom")

    a.consolidate = boom  # type: ignore[method-assign]

    results = await memory.consolidate_tenants()

    assert set(results) == {"A", "B"}
    assert results["A"][0].success is False
    assert "boom" in (results["A"][0].error or "")
    assert results["B"] and all(r.success for r in results["B"])


async def test_scoping_a_custom_domain_needs_a_factory():
    graph = MemoryGraph()
    embedder, llm = FakeEmbedder(), FakeLLM()

    class Wrapped:
        def __init__(self, semantic):
            self.semantic = semantic

        async def retrieve(self, *args, **kwargs):
            return await self.semantic.retrieve(*args, **kwargs)

        async def absorb(self, *args, **kwargs):
            return await self.semantic.absorb(*args, **kwargs)

    plain = CognitiveMemory(
        domain=Wrapped(SemanticMemory(graph, embedder, llm)),
        embedding_provider=embedder,
        llm_provider=llm,
        graph=graph,
    )
    with pytest.raises(ValueError, match="domain_factory"):
        plain.scoped("A")

    built: list = []

    def factory(g):
        built.append(g)
        return Wrapped(SemanticMemory(g, embedder, llm))

    with_factory = CognitiveMemory(
        domain=Wrapped(SemanticMemory(graph, embedder, llm)),
        embedding_provider=embedder,
        llm_provider=llm,
        graph=graph,
        domain_factory=factory,
    )
    scoped = with_factory.scoped("A")
    assert built and built[0].tenant == "A"
    assert scoped.tenant == "A"


# --- Forward migration ---------------------------------------------------------


def test_sqlite_store_without_tenant_column_migrates(tmp_path):
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE nodes (id TEXT PRIMARY KEY, node_type TEXT NOT NULL,
            content TEXT NOT NULL, embedding TEXT, utility_score REAL DEFAULT 1.0,
            access_count INTEGER DEFAULT 0, created_at TEXT NOT NULL,
            last_accessed TEXT NOT NULL, metadata TEXT DEFAULT '{}',
            data TEXT NOT NULL);
        CREATE TABLE edges (id TEXT PRIMARY KEY, source_id TEXT NOT NULL,
            target_id TEXT NOT NULL, relation_type TEXT NOT NULL,
            weight REAL DEFAULT 1.0, created_at TEXT NOT NULL,
            metadata TEXT DEFAULT '{}', data TEXT NOT NULL);
        INSERT INTO nodes (id, node_type, content, created_at, last_accessed, data)
        VALUES ('old', 'semantic', 'c', '2026-01-01T00:00:00+00:00',
                '2026-01-01T00:00:00+00:00',
                '{"id": "old", "node_type": "semantic", "content": "c",
                  "created_at": "2026-01-01T00:00:00+00:00",
                  "last_accessed": "2026-01-01T00:00:00+00:00"}');
        """
    )
    conn.commit()
    conn.close()

    backend = SQLiteBackend(path)
    try:
        # Pre-existing rows are untenanted: visible unscoped, hidden scoped.
        assert [n["id"] for n in backend.query_nodes(tenant=None)] == ["old"]
        assert backend.query_nodes(tenant="A") == []
        backend.save_node({**backend.load_node("old"), "id": "new", "tenant": "A"})
        assert [n["id"] for n in backend.query_nodes(tenant="A")] == ["new"]
    finally:
        backend.close()


def test_kuzu_store_without_tenant_column_migrates(tmp_path):
    import gc

    import kuzu

    path = str(tmp_path / "old_store")
    db = kuzu.Database(path)
    conn = kuzu.Connection(db)
    conn.execute(
        "CREATE NODE TABLE MemoryNode("
        "id STRING, node_type STRING, content STRING, embedding DOUBLE[3], "
        "utility_score DOUBLE, access_count INT64, created_at STRING, "
        "last_accessed STRING, metadata STRING, valid_from STRING, "
        "valid_until STRING, PRIMARY KEY(id))"
    )
    conn.execute(
        "CREATE REL TABLE MemoryEdge(FROM MemoryNode TO MemoryNode, id STRING, "
        "relation_type STRING, weight DOUBLE, created_at STRING, metadata STRING)"
    )
    conn.execute(
        "CREATE (:MemoryNode {id: 'old', node_type: 'semantic', content: 'c', "
        "utility_score: 1.0, access_count: 0, created_at: '2026-01-01', "
        "last_accessed: '2026-01-01', metadata: '{}'})"
    )
    del conn, db
    gc.collect()

    backend = KuzuBackend(path, embedding_dim=3)
    try:
        assert [n["id"] for n in backend.query_nodes(tenant=None)] == ["old"]
        assert backend.query_nodes(tenant="A") == []
    finally:
        backend.close()
