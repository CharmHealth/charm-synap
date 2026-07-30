"""Supersession is a durable tombstone, not an inferred edge (CH-728 item 5).

"Retired" must be recorded intrinsically on the retired node, not derived from
the presence of a `supersedes` edge. Deriving it from the edge means deleting
the superseder (by eviction or by hand) removes the edge and silently
resurrects the old version.

Procedural memory retires via a `superseded` metadata flag. Semantic memory
already retires via `valid_until` (an intrinsic marker) — these tests also pin
that its status reads no longer depend on the edge.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from synap.graph import MemoryGraph
from synap.procedural import ProceduralMemory
from synap.semantic import SemanticMemory
from synap.types import Procedure

from tests.conftest import FakeEmbedder, FakeLLM


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _proc(task_type: str, description: str, fields: list[str]) -> Procedure:
    return Procedure(
        task_type=task_type,
        description=description,
        schema={},
        field_ordering=fields,
    )


async def _register_v1_v2(graph: MemoryGraph, embedder: FakeEmbedder):
    proc = ProceduralMemory(graph=graph, embedding_provider=embedder)
    v1 = _proc("classify", "Classify items v1", ["evidence", "classification"])
    v2 = _proc("classify", "Classify items v2", ["evidence_for", "classification"])
    await proc.register(v1)
    await proc.register(v2)
    return proc, v1, v2


async def test_supersession_marks_old_version_and_keeps_lineage_edge():
    graph, embedder = MemoryGraph(), FakeEmbedder()
    proc, v1, v2 = await _register_v1_v2(graph, embedder)

    old = await graph.get_node(v1.id)
    assert old.metadata.get("superseded") is True, "retired version not tombstoned"
    # Lineage edge is retained (status and lineage are separate representations).
    assert await graph.has_incoming_edge(v1.id, "supersedes")
    assert (await proc.match("classify")).id == v2.id


async def test_deleting_superseder_does_not_resurrect_procedure():
    graph, embedder = MemoryGraph(), FakeEmbedder()
    proc, v1, v2 = await _register_v1_v2(graph, embedder)
    assert (await proc.match("classify")).id == v2.id

    # Delete the superseder — removes the supersedes edge via cascade.
    await graph.remove_node(v2.id)

    matched = await proc.match("classify")
    assert matched is None or matched.id != v1.id, (
        "deleting the superseder resurrected the retired procedure"
    )


async def test_register_supersession_is_atomic():
    """The new node, the retired flag, and the edge land together.

    A partial apply would leave two active versions or a retired version with no
    successor; write_batch makes it all-or-nothing.
    """
    graph, embedder = MemoryGraph(), FakeEmbedder()
    proc, v1, v2 = await _register_v1_v2(graph, embedder)

    # Exactly one active version for the task type.
    active = await proc.list_procedures(active_only=True)
    assert [p.id for p in active] == [v2.id]


# --- Semantic: durable via valid_until, and atomic supersession ---


async def _store_v1_v2(graph: MemoryGraph):
    sem = SemanticMemory(graph, FakeEmbedder(), FakeLLM())
    v1_id = await sem.store("Policy X is in effect")
    v2_id = await sem.store("Policy X is no longer in effect")  # -> SUPERSEDES
    return sem, v1_id, v2_id


async def test_semantic_superseded_fact_stays_retired_after_superseder_deleted():
    """Semantic retires via valid_until (intrinsic), so deleting the superseder
    does not resurrect the old fact — even though the edge is gone."""
    graph = MemoryGraph()
    sem, v1_id, v2_id = await _store_v1_v2(graph)

    v1 = await graph.get_node(v1_id)
    assert not await sem._is_current(v1, _now()), "old fact not retired"

    await graph.remove_node(v2_id)  # drop the superseder + its edge

    v1 = await graph.get_node(v1_id)
    assert not await sem._is_current(v1, _now()), "resurrected after superseder deleted"


async def test_semantic_supersession_applies_wholly():
    graph = MemoryGraph()
    sem, v1_id, v2_id = await _store_v1_v2(graph)

    v1 = await graph.get_node(v1_id)
    assert v1.valid_until is not None  # retired
    assert await graph.has_incoming_edge(v1_id, "supersedes")  # lineage kept
    v2 = await graph.get_node(v2_id)
    assert await sem._is_current(v2, _now())  # replacement is current


class _SupersessionWriteFails(MemoryGraph):
    """Fails the multi-node write (the supersession), leaving plain stores alone."""

    async def write_batch(self, nodes, edges):
        if len(nodes) > 1:
            raise RuntimeError("boom during supersession write")
        return await super().write_batch(nodes, edges)


async def test_semantic_supersession_is_atomic_on_write_failure():
    """If the supersession write fails, the old fact must not be left expired
    with no replacement — the whole transition rolls back."""
    graph = _SupersessionWriteFails()
    sem = SemanticMemory(graph, FakeEmbedder(), FakeLLM())
    v1_id = await sem.store("Policy X is in effect")  # plain store, one node

    with pytest.raises(RuntimeError):
        await sem.store("Policy X is no longer in effect")  # supersession -> fails

    v1 = await graph.get_node(v1_id)
    assert v1.valid_until is None, "old fact expired despite the write failing"
    assert await sem._is_current(v1, _now())
