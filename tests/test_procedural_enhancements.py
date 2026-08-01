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
