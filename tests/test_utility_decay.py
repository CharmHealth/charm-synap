"""Utility-score decay: recency gates frequency (CH-728 cluster 1, item 1).

The utility score fuses two signals — recency `r` (decays to ~0 when a node
goes cold) and a frequency proxy `f` (grows with access_count). The score must
satisfy three invariants:

  I1 — cold wins:  as r -> 0, score -> 0 regardless of f (no immortality floor).
  I2 — monotonic:  more recent and, independently, more frequent both raise it.
  I3 — fresh survives: f = 0 must not zero the score; a new, rarely-used node
       still gets its recency grace.

I1 + I3 force the form `score = r * (1 + f)` — recency multiplies, so it can
zero the score, while frequency enters through a factor floored at 1, so it can
boost but never gate. The previous formula added the two terms, which let any
node accessed twice (f >= 0.1) sit permanently at or above the 0.1 eviction
threshold — the additive floor. It also let `update_utility` decay from
`created_at` instead of `last_accessed`, so touching an old node scored it as
if idle. Both are fixed here.

The magnitude constants (`/20`, cap at f = 1) are inherited and untuned — the
invariants fix the structure, not the scale. Tuning them needs workload
telemetry we do not have yet; deferred.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from synap.graph import MemoryGraph
from synap.persistent_graph import compute_decay_score
from synap.types import MemoryNode, MemoryType

THRESHOLD = 0.1


def _ago(days: float) -> datetime:
    return datetime.now(UTC) - timedelta(days=days)


# --- canonical pure formula (persistent_graph.compute_decay_score) ---


def test_compute_decay_score_cold_popular_is_evictable():
    """I1: a node hit 100x but idle for months falls below threshold."""
    score = compute_decay_score(hours_since_access=60 * 24, access_count=100)
    assert score < THRESHOLD, "additive floor keeps a cold-but-popular node immortal"


def test_compute_decay_score_fresh_unpopular_survives():
    """I3: a just-accessed node with almost no history is not near eviction."""
    score = compute_decay_score(hours_since_access=0.0, access_count=1)
    assert score > 0.9


def test_compute_decay_score_frequency_still_boosts():
    """I2: at equal recency, more frequency means a higher score (not inert)."""
    warm_popular = compute_decay_score(hours_since_access=1.0, access_count=100)
    warm_rare = compute_decay_score(hours_since_access=1.0, access_count=0)
    assert warm_popular > warm_rare > 0.0


# --- MemoryGraph.decay_all + evict (in-memory formula sites) ---


async def test_evicts_cold_frequently_accessed_node():
    """I1 end-to-end: decay then evict removes a cold, often-used node."""
    g = MemoryGraph()
    node = MemoryNode(
        id="hot-then-cold",
        node_type=MemoryType.SEMANTIC,
        content="referenced heavily last spring, untouched since",
        access_count=50,
        created_at=_ago(60),
        last_accessed=_ago(60),
    )
    await g.add_node(node)

    await g.decay_all()
    evicted = await g.evict(threshold=THRESHOLD)

    assert "hot-then-cold" in evicted
    assert await g.get_node("hot-then-cold") is None


# --- the clock bug: update_utility must decay from last_accessed ---


async def test_accessing_an_old_node_does_not_evict_it():
    """Touching a memory records recency, so it must not make it evictable.

    `update_utility` decayed from `created_at`, so an old node scored ~0 the
    instant it was accessed — accessing a memory deleted it. It must decay from
    `last_accessed` (which `touch()` just set to now).
    """
    g = MemoryGraph()
    node = MemoryNode(
        id="old-but-hot",
        node_type=MemoryType.SEMANTIC,
        content="created long ago, accessed right now",
        access_count=0,
        created_at=_ago(60),
        last_accessed=_ago(60),
    )
    await g.add_node(node)

    await g.update_utility("old-but-hot")  # touch() sets last_accessed = now
    refreshed = await g.get_node("old-but-hot")
    assert refreshed is not None
    assert refreshed.utility_score > 0.9, "scored as idle despite being just accessed"

    evicted = await g.evict(threshold=THRESHOLD)
    assert "old-but-hot" not in evicted
    assert await g.get_node("old-but-hot") is not None


# --- Kuzu server-side Cypher copy of the formula must match ---


async def test_kuzu_decay_evicts_cold_popular_node(tmp_path):
    """The decay_all_scores Cypher is a fourth copy of the formula; pin it."""
    from synap.backends.kuzu import KuzuBackend
    from synap.persistent_graph import PersistentGraph

    backend = KuzuBackend(tmp_path / "g", embedding_dim=3)
    graph = PersistentGraph(backend=backend)
    backend.save_node(
        {
            "id": "cold-popular",
            "node_type": "semantic",
            "content": "hit 50x, idle for months",
            "embedding": [0.1, 0.2, 0.3],
            "utility_score": 1.0,
            "access_count": 50,
            "created_at": _ago(60).isoformat(),
            "last_accessed": _ago(60).isoformat(),
            "metadata": {},
        }
    )

    await graph.decay_all()
    evicted = await graph.evict(threshold=THRESHOLD)

    assert "cold-popular" in evicted
    assert backend.load_node("cold-popular") is None
    backend.close()
