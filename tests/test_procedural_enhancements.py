"""Procedural match prefers the most specific procedure (CH-728 follow-ups).

When several active procedures' task_types are substrings of a description, the
tiebreaker should be specificity (longest task_type), not whichever utility
order happened to surface first.
"""

from __future__ import annotations

from pathlib import Path

from synap.backends.sqlite import SQLiteBackend
from synap.graph import MemoryGraph
from synap.persistent_graph import PersistentGraph
from synap.procedural import ProceduralMemory
from synap.types import MemoryNode, MemoryType, Procedure
from tests.conftest import FakeEmbedder


def _proc(task_type: str) -> Procedure:
    return Procedure(
        task_type=task_type,
        description=f"procedure for {task_type}",
        schema={},
        field_ordering=["a", "b"],
    )


async def test_match_prefers_most_specific_task_type():
    graph = MemoryGraph()
    proc = ProceduralMemory(graph, FakeEmbedder())
    # Register the shorter (less specific) one first, so utility/insertion order
    # would surface it first without a specificity tiebreak.
    await proc.register(_proc("classify"))
    await proc.register(_proc("classify_urgent"))

    matched = await proc.match("please classify_urgent this case")

    assert matched is not None
    assert matched.task_type == "classify_urgent", (
        f"expected the most specific match, got {matched.task_type}"
    )


async def test_reconstructing_a_retired_version_does_not_hijack_the_index():
    """A cold-cache match reconstructs every matching version, retired ones
    included. Reconstructing a retired version must not repoint the task_type
    index at it — otherwise a later register() would supersede the retired node
    and leave two active versions."""
    graph = MemoryGraph()
    a = ProceduralMemory(graph, FakeEmbedder())
    v1, v2 = _proc("classify"), _proc("classify")
    await a.register(v1)
    await a.register(v2)  # v1 retired, v2 active

    b = ProceduralMemory(graph, FakeEmbedder())  # fresh instance (cold cache)
    await b._reconstruct_procedure(await graph.get_node(v2.id))  # index -> v2 (active)
    await b._reconstruct_procedure(await graph.get_node(v1.id))  # retired, reconstructed last

    assert b._task_type_index["classify"] == v2.id, "retired version hijacked the index"


async def test_cold_register_supersedes_the_version_already_in_the_graph():
    """register() reads the active version from the graph, not from its own index.

    A fresh instance — a process restart against persistent storage, a second
    instance sharing a graph, or any caller that hasn't run match() first — starts
    with an empty index. Trusting it would skip the tombstone and leave two nodes
    reading as active for one task_type, permanently: active procedures are
    exempt from eviction, so the duplicate never decays away.
    """
    graph = MemoryGraph()
    v1 = _proc("classify")
    await ProceduralMemory(graph, FakeEmbedder()).register(v1)

    cold = ProceduralMemory(graph, FakeEmbedder())  # empty index, no match() call
    v2 = _proc("classify")
    await cold.register(v2)

    active = [p.id for p in await cold.list_procedures(active_only=True)]
    assert active == [v2.id], f"expected only the new version active, got {active}"


async def test_register_retires_every_active_version_it_finds():
    """A store written before registration consulted the graph can already hold
    two active versions for one task_type. Registration retires all of them, so
    the duplicate is healed rather than left permanently un-evictable."""
    graph = MemoryGraph()
    proc = ProceduralMemory(graph, FakeEmbedder())
    for stale_id in ("stale-a", "stale-b"):
        await graph.add_node(
            MemoryNode(
                id=stale_id,
                content="classify: legacy",
                node_type=MemoryType.PROCEDURAL,
                metadata={
                    "task_type": "classify",
                    "description": "legacy",
                    "schema": {},
                    "field_ordering": ["a"],
                },
            )
        )

    v = _proc("classify")
    await proc.register(v)

    active = sorted(p.id for p in await proc.list_procedures(active_only=True))
    assert active == [v.id], f"expected only the new version active, got {active}"


async def test_register_after_restart_supersedes_across_persistent_storage(
    tmp_path: Path,
):
    """The same guarantee through a real backend, across a simulated restart:
    register the first version, close the store, reopen it, register a second
    version without any intervening read."""
    db = tmp_path / "procedures.db"
    v1, v2 = _proc("classify"), _proc("classify")

    backend = SQLiteBackend(db)
    await ProceduralMemory(PersistentGraph(backend=backend), FakeEmbedder()).register(v1)
    backend.close()

    backend = SQLiteBackend(db)
    reopened = ProceduralMemory(PersistentGraph(backend=backend), FakeEmbedder())
    await reopened.register(v2)
    try:
        active = [p.id for p in await reopened.list_procedures(active_only=True)]
        assert active == [v2.id], f"expected only the new version active, got {active}"
        assert (await reopened.match("please classify this")).id == v2.id
    finally:
        backend.close()
