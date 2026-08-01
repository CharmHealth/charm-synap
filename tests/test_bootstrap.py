"""Tests for the bootstrap cold start system."""

import json
from typing import Any

from synap._utils import safe_parse_json
from synap.bootstrap import Bootstrap
from synap.episodic import EpisodicMemory
from synap.graph import MemoryGraph
from synap.semantic import SemanticMemory
from synap.types import EpisodeOutcome, MemoryType
from tests.conftest import FakeEmbedder


class BootstrapLLM:
    """LLM that returns structured JSON for bootstrap operations."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate(
        self,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
    ) -> str:
        self.calls.append(prompt)

        if "Extract key facts" in prompt:
            return json.dumps({
                "nodes": [
                    {"content": "Step therapy required before surgery", "metadata": {"domain": "healthcare"}},
                    {"content": "Physical therapy is first-line treatment", "metadata": {"domain": "healthcare"}},
                    {"content": "Lumbar fusion is a surgical intervention", "metadata": {"domain": "healthcare"}},
                ],
                "edges": [
                    {"source": 2, "target": 0, "relation": "requires"},
                    {"source": 1, "target": 0, "relation": "part_of"},
                ],
            })

        if "Analyze this system prompt" in prompt:
            return json.dumps({
                "task_type": "prior_auth_determination",
                "description": "Determine prior authorization based on medical necessity",
                "field_ordering": ["diagnosis", "policy_criteria", "clinical_evidence", "determination"],
                "prerequisite_fields": {"determination": ["diagnosis", "policy_criteria", "clinical_evidence"]},
                "schema": {
                    "diagnosis": {"type": "string", "description": "Patient diagnosis"},
                    "policy_criteria": {"type": "string", "description": "Applicable policy criteria"},
                    "clinical_evidence": {"type": "string", "description": "Clinical evidence assessment"},
                    "determination": {"type": "string", "description": "Authorization determination"},
                },
            })

        return "{}"


def _make_bootstrap():
    graph = MemoryGraph()
    embedder = FakeEmbedder()
    llm = BootstrapLLM()
    semantic = SemanticMemory(graph=graph, embedding_provider=embedder)
    episodic = EpisodicMemory(graph=graph, embedding_provider=embedder)
    bootstrap = Bootstrap(
        semantic=semantic,
        episodic=episodic,
        embedding_provider=embedder,
        llm_provider=llm,
    )
    return bootstrap, graph, semantic, episodic, llm


async def test_extract_knowledge():
    bootstrap, graph, semantic, _, llm = _make_bootstrap()

    proposed = await bootstrap.extract_knowledge(
        texts=["Medical policy document about lumbar fusion..."],
        domain_hint="healthcare prior authorization",
    )

    assert len(proposed.nodes) == 3
    assert len(proposed.edges) == 2
    assert proposed.nodes[0].content == "Step therapy required before surgery"
    assert proposed.edges[0].relation_type == "requires"


async def test_extract_knowledge_summary():
    bootstrap, *_ = _make_bootstrap()

    proposed = await bootstrap.extract_knowledge(texts=["Some medical text"])
    summary = proposed.summary()

    assert "3 nodes" in summary
    assert "2 edges" in summary
    assert "Step therapy" in summary


async def test_accept_proposed_knowledge():
    bootstrap, graph, semantic, _, _ = _make_bootstrap()

    proposed = await bootstrap.extract_knowledge(texts=["Medical policy document"])
    node_ids = await bootstrap.accept(proposed)

    assert len(node_ids) == 3
    assert await graph.node_count(MemoryType.SEMANTIC) == 3
    assert await graph.edge_count() >= 2


async def test_infer_procedure():
    bootstrap, *_ = _make_bootstrap()

    procedure = await bootstrap.infer_procedure(
        system_prompt="You are a medical reviewer. Assess clinical evidence...",
        example_outputs=[{"determination": "approved", "diagnosis": "M54.5"}],
    )

    assert procedure.task_type == "prior_auth_determination"
    assert "determination" in procedure.field_ordering
    assert procedure.prerequisite_fields.get("determination") is not None
    assert len(procedure.schema) > 0


async def test_ingest_logs():
    bootstrap, graph, _, episodic, _ = _make_bootstrap()

    logs = [
        {
            "input": "Diagnose TypeError in webhook",
            "output": {"fix": "add null check"},
            "outcome": "success",
            "task_type": "diagnose_bug",
        },
        {
            "input": "Diagnose connection timeout",
            "output": {"fix": "wrong diagnosis"},
            "outcome": "failure",
            "task_type": "diagnose_bug",
        },
        {
            "cue": "Missing field validation",
            "content": {"fix": "add validation"},
            "outcome": "corrected",
            "correction": "Should validate at API boundary",
        },
    ]

    episodes = await bootstrap.ingest_logs(logs, task_type="diagnose_bug")
    assert len(episodes) == 3
    assert await episodic.episode_count() == 3
    assert episodes[0].outcome == EpisodeOutcome.SUCCESS
    assert episodes[1].outcome == EpisodeOutcome.FAILURE
    assert episodes[2].outcome == EpisodeOutcome.CORRECTED
    assert episodes[2].correction == "Should validate at API boundary"


async def test_full_bootstrap_flow():
    """End-to-end: extract, review, accept, then use."""
    bootstrap, graph, semantic, _, _ = _make_bootstrap()

    proposed = await bootstrap.extract_knowledge(
        texts=["Payer policies for orthopedic procedures"],
        domain_hint="healthcare",
    )

    assert len(proposed.nodes) > 0

    await bootstrap.accept(proposed)

    result = await semantic.search("step therapy")
    assert len(result.nodes) > 0


# --- JSON parsing edge cases ---


def testsafe_parse_json_direct():
    assert safe_parse_json('{"a": 1}') == {"a": 1}


def testsafe_parse_json_markdown_block():
    text = '```json\n{"a": 1}\n```'
    assert safe_parse_json(text) == {"a": 1}


def testsafe_parse_json_with_preamble():
    text = 'Here is the result:\n{"a": 1}'
    assert safe_parse_json(text) == {"a": 1}


def testsafe_parse_json_invalid():
    assert safe_parse_json("not json at all") is None
