"""Procedural match prefers the most specific procedure (CH-728 follow-ups).

When several active procedures' task_types are substrings of a description, the
tiebreaker should be specificity (longest task_type), not whichever utility
order happened to surface first.
"""

from __future__ import annotations

from synap.graph import MemoryGraph
from synap.procedural import ProceduralMemory
from synap.types import Procedure
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
