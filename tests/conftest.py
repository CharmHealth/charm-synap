"""Shared test fixtures — fake providers for testing without external deps."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from synap.graph import MemoryGraph


class FakeEmbedder:
    """Deterministic embedder for testing. Produces a simple hash-based vector."""

    async def embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        # 8-dimensional vector from hash bytes
        return [b / 255.0 for b in h[:8]]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class FakeLLM:
    """Fake LLM that returns canned responses for testing."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate(
        self,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
    ) -> str:
        self.calls.append(prompt)
        # Return a reasonable consolidation response
        if "Extract the key facts" in prompt:
            return "Consolidated fact from multiple episodes."
        if "Propose ONE new check or reasoning field" in prompt:
            return '{"field_name": "verification_check", "field_type": "string", "field_description": "Verify all required data is present before proceeding", "insert_before": "determination"}'
        if "extract" in prompt.lower() and "conversation" in prompt.lower():
            return '{"facts": ["Aetna requires step therapy before lumbar fusion", "Patient completed 6 weeks of PT"], "summary": "Agent checked prior auth requirements and confirmed approval", "input_summary": "Check prior auth for knee replacement"}'
        if "SUPERSEDES or COEXISTS" in prompt:
            # If the prompt contains "discontinued" or "no longer", it's a supersession
            if "discontinued" in prompt.lower() or "no longer" in prompt.lower():
                return "SUPERSEDES"
            return "COEXISTS"
        if "Merge them into a single" in prompt:
            return "Merged semantic fact."
        return "LLM response."


@pytest.fixture
def graph() -> MemoryGraph:
    return MemoryGraph()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()
