"""Tests for the agent-callable memory tools module."""

from synap.facade import CognitiveMemory
from synap.graph import MemoryGraph
from synap.semantic import SemanticMemory
from synap.tools import memory_tools
from synap.types import Procedure
from tests.conftest import FakeEmbedder, FakeLLM


def _make_memory() -> CognitiveMemory:
    graph = MemoryGraph()
    embedder = FakeEmbedder()
    domain = SemanticMemory(graph=graph, embedding_provider=embedder)
    return CognitiveMemory(
        domain=domain,
        embedding_provider=embedder,
        llm_provider=FakeLLM(),
        graph=graph,
    )


def _get_tool(tools, name):
    return next(t for t in tools if t["name"] == name)


async def test_remember_fact_stores_in_semantic():
    memory = _make_memory()
    tools = memory_tools(memory)
    handler = _get_tool(tools, "remember_fact")["handler"]

    result = await handler({"content": "Aetna requires step therapy"})
    assert result["stored"] is True
    assert result["node_id"] is not None

    stats = await memory.stats()
    assert stats.semantic_nodes >= 1


async def test_recall_searches_semantic_and_episodic():
    memory = _make_memory()
    tools = memory_tools(memory)

    # Store a fact
    await _get_tool(tools, "remember_fact")["handler"](
        {"content": "Step therapy required before surgery"}
    )

    # Record an episode
    await _get_tool(tools, "record_observation")["handler"]({
        "description": "Checked step therapy requirements",
        "output": {"result": "approved"},
        "outcome": "success",
        "task_type": "prior_auth",
    })

    # Recall should find both
    result = await _get_tool(tools, "recall")["handler"](
        {"query": "step therapy"}
    )
    assert "facts" in result
    assert "episodes" in result


async def test_recall_filters_by_memory_type():
    memory = _make_memory()
    tools = memory_tools(memory)

    await _get_tool(tools, "remember_fact")["handler"](
        {"content": "Test fact"}
    )

    # Semantic only
    result = await _get_tool(tools, "recall")["handler"](
        {"query": "test", "memory_type": "semantic"}
    )
    assert "facts" in result
    assert "episodes" not in result

    # Episodic only
    result = await _get_tool(tools, "recall")["handler"](
        {"query": "test", "memory_type": "episodic"}
    )
    assert "episodes" in result
    assert "facts" not in result


async def test_record_observation_creates_episode():
    memory = _make_memory()
    tools = memory_tools(memory)
    handler = _get_tool(tools, "record_observation")["handler"]

    result = await handler({
        "description": "Diagnosed webhook error",
        "output": {"diagnosis": "null ref"},
        "outcome": "failure",
        "task_type": "diagnose_bug",
    })
    assert result["recorded"] is True

    stats = await memory.stats()
    assert stats.total_episodes == 1


async def test_suggest_amendment_queues_proposal():
    memory = _make_memory()
    tools = memory_tools(memory)
    handler = _get_tool(tools, "suggest_amendment")["handler"]

    result = await handler({
        "task_type": "prior_auth",
        "suggestion": "Add step therapy verification before determination",
        "rationale": "Missed in 2 recent cases",
    })
    assert result["queued"] is True
    assert result["proposal_id"] is not None

    # Proposal is stored as a semantic node
    node = await memory.graph.get_node(result["proposal_id"])
    assert node is not None
    assert node.metadata["type"] == "amendment_proposal"
    assert node.metadata["task_type"] == "prior_auth"


async def test_suggest_amendment_does_not_modify_procedure():
    """Suggestions are queued, not applied directly."""
    memory = _make_memory()
    tools = memory_tools(memory)

    await memory.procedural.register(Procedure(
        task_type="prior_auth",
        description="Prior authorization",
        schema={"determination": {"type": "string"}},
        field_ordering=["evidence", "determination"],
        prerequisite_fields={"determination": ["evidence"]},
    ))

    await _get_tool(tools, "suggest_amendment")["handler"]({
        "task_type": "prior_auth",
        "suggestion": "Remove evidence step",
    })

    # Procedure should be unchanged
    proc = await memory.procedural.match("prior_auth", task_type="prior_auth")
    assert proc.field_ordering == ["evidence", "determination"]


async def test_get_procedure_returns_active():
    memory = _make_memory()
    tools = memory_tools(memory)

    await memory.procedural.register(Procedure(
        task_type="diagnose_bug",
        description="Diagnose bugs",
        schema={"root_cause": {"type": "string"}, "fix": {"type": "string"}},
        field_ordering=["root_cause", "fix"],
        prerequisite_fields={"fix": ["root_cause"]},
    ))

    result = await _get_tool(tools, "get_procedure")["handler"](
        {"task_type": "diagnose_bug"}
    )
    assert result["found"] is True
    assert result["task_type"] == "diagnose_bug"
    assert result["field_ordering"] == ["root_cause", "fix"]


async def test_get_procedure_not_found():
    memory = _make_memory()
    tools = memory_tools(memory)

    result = await _get_tool(tools, "get_procedure")["handler"](
        {"task_type": "nonexistent"}
    )
    assert result["found"] is False


async def test_tool_schemas_are_valid():
    """All tools have name, description, schema, and handler."""
    memory = _make_memory()
    tools = memory_tools(memory)

    assert len(tools) == 5
    for tool in tools:
        assert "name" in tool
        assert "description" in tool
        assert "schema" in tool
        assert "handler" in tool
        assert callable(tool["handler"])
        assert tool["schema"]["type"] == "object"
