"""Tests for the three memory subsystems."""

from datetime import UTC

from synap.episodic import EpisodicMemory
from synap.graph import MemoryGraph
from synap.procedural import ProceduralMemory
from synap.semantic import SemanticMemory
from synap.types import Episode, EpisodeOutcome, MemoryType, Procedure, ToolCall
from tests.conftest import FakeEmbedder, FakeLLM

# --- Semantic Memory ---


async def test_semantic_store_and_search(graph: MemoryGraph, embedder: FakeEmbedder):
    sem = SemanticMemory(graph=graph, embedding_provider=embedder)

    node_id = await sem.store("Aetna requires step therapy before lumbar fusion")
    assert await graph.get_node(node_id) is not None
    assert (await graph.get_node(node_id)).node_type == MemoryType.SEMANTIC

    result = await sem.search("step therapy lumbar fusion")
    assert len(result.nodes) > 0
    assert any("step therapy" in n.content for n in result.nodes)


async def test_semantic_link_and_traverse(graph: MemoryGraph, embedder: FakeEmbedder):
    sem = SemanticMemory(graph=graph, embedding_provider=embedder)

    id1 = await sem.store("Physical therapy is conservative treatment")
    id2 = await sem.store("Lumbar fusion requires prior auth")
    await sem.link(id2, id1, "step_therapy_before")

    result = await sem.search("lumbar fusion")
    assert len(result.nodes) >= 1


async def test_semantic_summary(graph: MemoryGraph, embedder: FakeEmbedder):
    sem = SemanticMemory(graph=graph, embedding_provider=embedder)
    await sem.store("Fact one")
    await sem.store("Fact two")

    result = await sem.search("facts")
    summary = result.summary
    assert "Fact one" in summary or "Fact two" in summary


async def test_semantic_domain_retrieve(graph: MemoryGraph, embedder: FakeEmbedder):
    """SemanticMemory implements SemanticDomain.retrieve."""
    sem = SemanticMemory(graph=graph, embedding_provider=embedder)
    await sem.store("Step therapy required before surgery")

    results = await sem.retrieve("step therapy")
    assert len(results) > 0
    assert any("step therapy" in r.content.lower() for r in results)
    assert all(hasattr(r, "relevance") for r in results)


async def test_semantic_domain_absorb(graph: MemoryGraph, embedder: FakeEmbedder):
    """SemanticMemory implements SemanticDomain.absorb."""
    from synap.types import MemoryNode

    sem = SemanticMemory(graph=graph, embedding_provider=embedder)
    episode = MemoryNode(content="test episode", node_type=MemoryType.EPISODIC)

    node_id = await sem.absorb(
        insights=["Consolidated insight from episodes"],
        source_episodes=[episode],
        metadata={"task_type": "test"},
    )

    assert node_id is not None
    assert await graph.get_node(node_id) is not None


async def test_semantic_contradiction_detection(graph: MemoryGraph, embedder: FakeEmbedder):
    """Storing a contradicting fact supersedes the old one."""
    llm = FakeLLM()
    sem = SemanticMemory(graph=graph, embedding_provider=embedder, llm_provider=llm)

    old_id = await sem.store("Patient takes metformin daily")
    new_id = await sem.store("Patient discontinued metformin")

    # Old node should have a supersedes edge pointing to it
    assert await graph.has_incoming_edge(old_id, "supersedes")

    # Old node should have valid_until set
    old_node = await graph.get_node(old_id)
    assert old_node.valid_until is not None

    # New node should have valid_from set
    new_node = await graph.get_node(new_id)
    assert new_node.valid_from is not None


async def test_semantic_search_excludes_superseded(graph: MemoryGraph, embedder: FakeEmbedder):
    """Search results exclude superseded facts."""
    llm = FakeLLM()
    sem = SemanticMemory(graph=graph, embedding_provider=embedder, llm_provider=llm)

    await sem.store("Patient takes metformin daily")
    await sem.store("Patient discontinued metformin")

    result = await sem.search("metformin")
    # Only the current fact should appear
    contents = [n.content for n in result.nodes]
    assert "Patient discontinued metformin" in contents
    assert "Patient takes metformin daily" not in contents


async def test_semantic_coexisting_facts(graph: MemoryGraph, embedder: FakeEmbedder):
    """Non-contradicting facts coexist without supersession."""
    llm = FakeLLM()
    sem = SemanticMemory(graph=graph, embedding_provider=embedder, llm_provider=llm)

    id1 = await sem.store("Patient takes metformin daily")
    id2 = await sem.store("Patient allergic to penicillin")

    # Neither should be superseded
    assert not await graph.has_incoming_edge(id1, "supersedes")
    assert not await graph.has_incoming_edge(id2, "supersedes")


async def test_semantic_store_skip_contradictions(graph: MemoryGraph, embedder: FakeEmbedder):
    """check_contradictions=False skips detection (for bulk import)."""
    llm = FakeLLM()
    sem = SemanticMemory(graph=graph, embedding_provider=embedder, llm_provider=llm)

    id1 = await sem.store("Patient takes metformin daily")
    await sem.store(
        "Patient discontinued metformin",
        check_contradictions=False,
    )

    # No supersession because detection was skipped
    assert not await graph.has_incoming_edge(id1, "supersedes")
    # No LLM calls for contradiction check
    contradiction_calls = [c for c in llm.calls if "SUPERSEDES or COEXISTS" in c]
    assert len(contradiction_calls) == 0


async def test_semantic_no_llm_skips_contradictions(graph: MemoryGraph, embedder: FakeEmbedder):
    """Without an LLM provider, contradiction detection is silently skipped."""
    sem = SemanticMemory(graph=graph, embedding_provider=embedder)  # no LLM

    id1 = await sem.store("Patient takes metformin daily")
    await sem.store("Patient discontinued metformin")

    # No supersession because no LLM to detect it
    assert not await graph.has_incoming_edge(id1, "supersedes")


async def test_semantic_retrieve_filters_expired(graph: MemoryGraph, embedder: FakeEmbedder):
    """Retrieve filters out nodes past their valid_until date."""
    from datetime import datetime, timedelta

    from synap.types import MemoryNode

    sem = SemanticMemory(graph=graph, embedding_provider=embedder)

    # Manually create an expired node
    past = datetime.now(UTC) - timedelta(days=30)
    expired_node = MemoryNode(
        content="Expired insurance coverage",
        node_type=MemoryType.SEMANTIC,
        embedding=await embedder.embed("Expired insurance coverage"),
        valid_from=past - timedelta(days=365),
        valid_until=past,
    )
    await graph.add_node(expired_node)

    # Create a current node
    await sem.store("Active insurance coverage", check_contradictions=False)

    result = await sem.search("insurance coverage")
    contents = [n.content for n in result.nodes]
    assert "Active insurance coverage" in contents
    assert "Expired insurance coverage" not in contents


# --- Procedural Memory ---


async def test_procedural_register_and_match(graph: MemoryGraph, embedder: FakeEmbedder):
    proc = ProceduralMemory(graph=graph, embedding_provider=embedder)

    procedure = Procedure(
        task_type="prior_auth_determination",
        description="Determine prior authorization for medical services",
        schema={"determination": {"type": "string"}},
        field_ordering=["clinical_evidence", "reasoning", "determination"],
        prerequisite_fields={"determination": ["clinical_evidence", "reasoning"]},
    )
    await proc.register(procedure)

    matched = await proc.match("Determine prior authorization for lumbar fusion")
    assert matched is not None
    assert matched.task_type == "prior_auth_determination"


async def test_procedural_build_schema_enforces_ordering(
    graph: MemoryGraph, embedder: FakeEmbedder
):
    proc = ProceduralMemory(graph=graph, embedding_provider=embedder)

    procedure = Procedure(
        task_type="diagnose_bug",
        description="Diagnose a bug from error logs",
        schema={
            "error_classification": {"type": "string"},
            "root_cause": {"type": "string"},
            "fix_proposal": {"type": "string"},
        },
        field_ordering=["error_classification", "root_cause", "fix_proposal"],
        prerequisite_fields={"fix_proposal": ["error_classification", "root_cause"]},
    )
    await proc.register(procedure)

    schema = await proc.build_schema(procedure)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["error_classification", "root_cause", "fix_proposal"]


async def test_procedural_versioning(graph: MemoryGraph, embedder: FakeEmbedder):
    proc = ProceduralMemory(graph=graph, embedding_provider=embedder)

    v1 = Procedure(
        task_type="classify",
        description="Classify items v1",
        schema={},
        field_ordering=["evidence", "classification"],
    )
    v2 = Procedure(
        task_type="classify",
        description="Classify items v2 with improved evidence gathering",
        schema={},
        field_ordering=["evidence_for", "evidence_against", "classification"],
    )

    await proc.register(v1)
    await proc.register(v2)

    assert await graph.has_incoming_edge(v1.id, "supersedes")

    # Match should return v2 (async filtering)
    matched = await proc.match("classify")
    assert matched is not None
    assert matched.id == v2.id


async def test_procedural_match_reconstructs_from_graph(
    graph: MemoryGraph, embedder: FakeEmbedder
):
    """Match reconstructs procedures from graph after session cache is cleared."""
    proc = ProceduralMemory(graph=graph, embedding_provider=embedder)

    procedure = Procedure(
        task_type="diagnose_bug",
        description="Diagnose a bug from error logs",
        schema={
            "error_classification": {"type": "string"},
            "root_cause": {"type": "string"},
            "fix_proposal": {"type": "string"},
        },
        field_ordering=["error_classification", "root_cause", "fix_proposal"],
        prerequisite_fields={"fix_proposal": ["error_classification", "root_cause"]},
        system_prompt_fragment="Analyze systematically.",
    )
    await proc.register(procedure)

    # Clear session cache — simulates process restart
    proc._procedures.clear()
    proc._task_type_index.clear()

    matched = await proc.match("Diagnose a bug in the webhook handler")
    assert matched is not None
    assert matched.task_type == "diagnose_bug"
    assert matched.field_ordering == ["error_classification", "root_cause", "fix_proposal"]
    assert matched.prerequisite_fields == {"fix_proposal": ["error_classification", "root_cause"]}
    assert matched.system_prompt_fragment == "Analyze systematically."
    assert matched.schema == procedure.schema


# --- Episodic Memory ---


async def test_episodic_record_and_recall(graph: MemoryGraph, embedder: FakeEmbedder):
    ep = EpisodicMemory(graph=graph, embedding_provider=embedder)

    episode = Episode(
        cue="TypeError in stripe webhook handler",
        content={"error": "Cannot read property 'amount'", "fix": "add null check"},
        outcome=EpisodeOutcome.SUCCESS,
        task_type="diagnose_bug",
    )
    await ep.record(episode)
    assert await ep.episode_count() == 1
    assert await graph.node_count(MemoryType.EPISODIC) == 3

    recalled = await ep.recall("TypeError in webhook handler")
    assert len(recalled) == 1
    assert recalled[0].id == episode.id


async def test_episodic_prioritizes_failures(graph: MemoryGraph, embedder: FakeEmbedder):
    ep = EpisodicMemory(graph=graph, embedding_provider=embedder)

    success = Episode(
        cue="payment processing error",
        content={"result": "fixed"},
        outcome=EpisodeOutcome.SUCCESS,
        task_type="diagnose_bug",
    )
    failure = Episode(
        cue="payment processing error",
        content={"result": "wrong diagnosis"},
        outcome=EpisodeOutcome.FAILURE,
        task_type="diagnose_bug",
    )
    await ep.record(success)
    await ep.record(failure)

    recalled = await ep.recall("payment processing error", max_episodes=1)
    assert len(recalled) == 1
    assert recalled[0].outcome == EpisodeOutcome.FAILURE


async def test_episodic_find_patterns(graph: MemoryGraph, embedder: FakeEmbedder):
    ep = EpisodicMemory(graph=graph, embedding_provider=embedder)

    for i in range(3):
        await ep.record(
            Episode(
                cue=f"auth failure case {i}",
                content={"error": f"error_{i}"},
                outcome=EpisodeOutcome.FAILURE,
                task_type="prior_auth",
            )
        )

    patterns = await ep.find_patterns("prior_auth", min_occurrences=3)
    assert len(patterns) == 1
    assert patterns[0].outcome == EpisodeOutcome.FAILURE
    assert patterns[0].occurrences == 3


async def test_episodic_generate_warnings(graph: MemoryGraph, embedder: FakeEmbedder):
    ep = EpisodicMemory(graph=graph, embedding_provider=embedder)

    failure = Episode(
        cue="missing PT documentation",
        content={"error": "denied"},
        outcome=EpisodeOutcome.FAILURE,
        task_type="prior_auth",
    )
    corrected = Episode(
        cue="wrong billing code",
        content={"code": "99213"},
        outcome=EpisodeOutcome.CORRECTED,
        correction="Should have been 99214",
        task_type="billing",
    )

    warnings = ep.generate_warnings([failure, corrected])
    assert len(warnings) == 2
    assert "failure" in warnings[0].lower() or "Previous" in warnings[0]
    assert "99214" in warnings[1]


async def test_episodic_recall_reconstructs_from_graph(
    graph: MemoryGraph, embedder: FakeEmbedder
):
    """Recall reconstructs episodes from graph after session cache is cleared."""
    ep = EpisodicMemory(graph=graph, embedding_provider=embedder)

    episode = Episode(
        cue="TypeError in stripe webhook handler",
        content={"error": "Cannot read property 'amount'", "fix": "add null check"},
        outcome=EpisodeOutcome.SUCCESS,
        task_type="diagnose_bug",
        tags=["stripe", "webhook"],
    )
    await ep.record(episode)

    # Clear session cache — simulates process restart
    ep._episodes.clear()

    recalled = await ep.recall("TypeError in webhook handler")
    assert len(recalled) == 1
    assert recalled[0].id == episode.id
    assert recalled[0].cue == episode.cue
    assert recalled[0].content == episode.content
    assert recalled[0].outcome == episode.outcome
    assert recalled[0].task_type == episode.task_type
    assert recalled[0].tags == episode.tags


async def test_episodic_recall_reconstructs_with_tool_calls(
    graph: MemoryGraph, embedder: FakeEmbedder
):
    """Recall reconstructs tool calls from graph metadata."""
    ep = EpisodicMemory(graph=graph, embedding_provider=embedder)

    episode = Episode(
        cue="Find the webhook handler source",
        content={"result": "found handler"},
        outcome=EpisodeOutcome.SUCCESS,
        task_type="code_search",
        tool_calls=[
            ToolCall(
                query="find webhook handler",
                server="code-search",
                tool_name="search_files",
                parameters={"pattern": "handleWebhook"},
                result_summary="Found src/webhooks/stripe.ts:45",
                success=True,
            ),
        ],
    )
    await ep.record(episode)

    # Clear session cache
    ep._episodes.clear()

    recalled = await ep.recall("webhook handler source")
    assert len(recalled) == 1
    assert len(recalled[0].tool_calls) == 1
    tc = recalled[0].tool_calls[0]
    assert tc.tool_name == "search_files"
    assert tc.server == "code-search"
    assert tc.success is True


async def test_episodic_recall_reconstructs_correction(
    graph: MemoryGraph, embedder: FakeEmbedder
):
    """Recall reconstructs correction field from outcome node metadata."""
    ep = EpisodicMemory(graph=graph, embedding_provider=embedder)

    episode = Episode(
        cue="wrong billing code used",
        content={"code": "99213"},
        outcome=EpisodeOutcome.CORRECTED,
        correction="Should have been 99214",
        task_type="billing",
    )
    await ep.record(episode)

    # Clear session cache
    ep._episodes.clear()

    recalled = await ep.recall("billing code")
    assert len(recalled) == 1
    assert recalled[0].correction == "Should have been 99214"
    assert recalled[0].outcome == EpisodeOutcome.CORRECTED
