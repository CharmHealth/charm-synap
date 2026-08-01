"""Built-in placeholder providers for offline and demo use.

These let ``synap.mcp_server`` and the quickstarts run with no external services.
They are deliberately NOT semantic: ``HashEmbedder`` derives vectors from a text
hash (so similarity is meaningless) and ``PlaceholderLLM`` returns canned text
without reasoning. Configure a real embedding model and LLM for production —
wiring those from an environment variable is tracked separately.
"""

from __future__ import annotations

import hashlib


class HashEmbedder:
    """Deterministic, dependency-free embedder for offline/demo/testing.

    Produces reproducible vectors of the requested dimension from a hash of the
    text. There is no semantic meaning to the distances — swap in a real
    embedding model for meaningful similarity search.
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    async def embed(self, text: str) -> list[float]:
        out: list[float] = []
        counter = 0
        data = text.encode()
        while len(out) < self._dim:
            digest = hashlib.sha256(data + counter.to_bytes(4, "big")).digest()
            out.extend(b / 255.0 for b in digest)
            counter += 1
        return out[: self._dim]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class PlaceholderLLM:
    """LLM stand-in that does no reasoning, so the pipeline runs without a model.

    Returns safe, generic output; in particular it never asserts a supersession
    on its own. Configure a real LLM for meaningful consolidation and
    contradiction detection.
    """

    async def generate(
        self, prompt: str, output_schema: dict | None = None
    ) -> str:
        if "SUPERSEDES or COEXISTS" in prompt:
            return "COEXISTS"  # never auto-supersede without a real judgment
        return "Placeholder consolidation output."
