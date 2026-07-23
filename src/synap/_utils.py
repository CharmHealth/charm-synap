"""Shared utility functions."""

from __future__ import annotations

import json
from typing import Any


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors.

    Raises ValueError on a dimension mismatch rather than returning 0.0: a
    length mismatch is a configuration bug (embeddings from different models or
    a truncated store), and silently scoring it as zero hides that bug behind a
    plausible-looking ranking.
    """
    if len(a) != len(b):
        raise ValueError(
            f"embedding dimension mismatch: {len(a)} vs {len(b)}"
        )
    if len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def select_evictions(
    items: list[tuple[str, float, str | None]], threshold: float
) -> list[str]:
    """Decide which node ids to evict, treating an episode as one unit.

    ``items`` is ``(node_id, utility_score, episode_id_or_None)``. A node with no
    ``episode_id`` is evicted individually when its score is below ``threshold``.
    An episode's nodes are evicted only when *every* member is below threshold —
    "whole episode or nothing" — so decay of one node can't shear the episode
    into unreadable orphans (policy B1).
    """
    victims: list[str] = []
    episodes: dict[str, list[tuple[str, float]]] = {}
    for node_id, score, episode_id in items:
        if not episode_id:  # None or "" -> a standalone (non-episodic) node
            if score < threshold:
                victims.append(node_id)
        else:
            episodes.setdefault(episode_id, []).append((node_id, score))
    for members in episodes.values():
        if all(score < threshold for _, score in members):
            victims.extend(node_id for node_id, _ in members)
    return victims


def safe_parse_json(text: str) -> dict[str, Any] | None:
    """Parse JSON from LLM output, handling common formatting issues."""
    text = text.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    if "```" in text:
        start = text.find("```")
        start = text.find("\n", start) + 1
        end = text.find("```", start)
        if end > start:
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass

    # Try finding first { to last }
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            pass

    return None
