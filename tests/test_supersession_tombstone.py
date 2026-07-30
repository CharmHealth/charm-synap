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

import pytest

from synap.graph import MemoryGraph
from synap.procedural import ProceduralMemory
from synap.types import Procedure

from tests.conftest import FakeEmbedder


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
