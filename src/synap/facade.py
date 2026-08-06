"""CognitiveMemory — the primary integration interface."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from synap._utils import safe_parse_json
from synap.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationResult,
)
from synap.episodic import EpisodicMemory
from synap.graph import MemoryGraph
from synap.persistent_graph import PersistentGraph
from synap.procedural import ProceduralMemory
from synap.protocols import (
    EmbeddingProvider,
    GraphStore,
    LLMProvider,
    SemanticDomain,
    StorageBackend,
)
from synap.types import (
    CapacityHints,
    DomainResult,
    Episode,
    EpisodeOutcome,
    MemoryType,
    PreparedContext,
    ToolCall,
)

_EXTRACT_PROMPT = """\
Analyze this conversation and extract:
1. Key facts worth remembering long-term (not ephemeral details)
2. A summary of what the agent did and what happened

Conversation:
{conversation}

Respond with JSON:
{{
  "facts": ["fact 1", "fact 2", ...],
  "summary": "one-sentence summary of what happened",
  "input_summary": "what the user asked for or what triggered this"
}}"""


@dataclass
class MemoryStats:
    """Current state of the memory system."""

    semantic_nodes: int = 0
    procedural_nodes: int = 0
    episodic_nodes: int = 0
    total_edges: int = 0
    total_episodes: int = 0
    pending_consolidation: int = 0


@dataclass
class EvaluationReport:
    """Evaluation metrics from episodic outcome data."""

    outcome_trend: dict[str, list[float]] = field(default_factory=dict)
    retrieval_hit_rate: float = 0.0
    warning_effectiveness: float = 0.0
    cold_spots: list[str] = field(default_factory=list)
    hot_spots: list[str] = field(default_factory=list)


class CognitiveMemory:
    """Main entry point for the synap memory library.

    Consumers use this to store experiences, retrieve context, and
    get output schemas for LLM calls. The library prepares context
    and records outcomes; the consumer owns the agent loop and LLM client.
    """

    def __init__(
        self,
        domain: SemanticDomain,
        embedding_provider: EmbeddingProvider,
        llm_provider: LLMProvider,
        graph: GraphStore | None = None,
        capacity: CapacityHints | None = None,
        consolidation_config: ConsolidationConfig | None = None,
        utility_decay_rate: float = 0.01,
        backend: StorageBackend | None = None,
    ) -> None:
        self._capacity = capacity or CapacityHints()
        self._llm = llm_provider

        if graph is not None:
            self._graph = graph
        elif backend is not None:
            self._graph = PersistentGraph(
                backend=backend,
                utility_decay_rate=utility_decay_rate,
            )
        else:
            self._graph = MemoryGraph(utility_decay_rate=utility_decay_rate)

        # Guard against split-graph misconfiguration
        domain_graph = getattr(domain, "_graph", None)
        if domain_graph is not None and domain_graph is not self._graph:
            raise ValueError(
                "domain and CognitiveMemory must share the same graph instance. "
                "Pass the same graph object to both SemanticMemory and CognitiveMemory."
            )

        self._domain = domain
        self._procedural = ProceduralMemory(
            graph=self._graph,
            embedding_provider=embedding_provider,
        )
        self._episodic = EpisodicMemory(
            graph=self._graph,
            embedding_provider=embedding_provider,
        )
        self._consolidation = ConsolidationEngine(
            graph=self._graph,
            domain=self._domain,
            procedural=self._procedural,
            episodic=self._episodic,
            llm_provider=llm_provider,
            config=consolidation_config or ConsolidationConfig(),
        )

        # Session-scoped operational metrics (reset on restart — not memory)
        self._retrieval_attempts = 0
        self._retrieval_hits = 0
        self._warned_calls = 0
        self._warnings_followed_by_success = 0
        self._last_call_had_warnings = False

    # --- High-level operations ---

    async def prepare_call(
        self,
        task_description: str,
        task_type: str | None = None,
        input_data: dict[str, Any] | None = None,
    ) -> PreparedContext:
        """Prepare everything needed for an LLM call.

        1. Match task to a procedure (procedural memory)
        2. Build output schema with enforced field ordering
        3. Retrieve domain knowledge (via SemanticDomain adapter)
        4. Recall relevant episodes (reconstructive retrieval)
        5. Package as PreparedContext
        """
        self._retrieval_attempts += 1

        # 1. Procedural: match and build schema
        procedure = await self._procedural.match(task_description, task_type=task_type)
        output_schema = None
        prompt_fragment = None

        proc_task_type = procedure.task_type if procedure else task_type

        # 3. Episodic: recall relevant past experiences
        episodes = await self._episodic.recall(
            cue=task_description,
            task_type=proc_task_type,
            max_episodes=3,
            capacity=self._capacity,
        )
        warnings = self._episodic.generate_warnings(episodes)

        # Build few-shot examples from successful episodes
        few_shot = None
        successful = [e for e in episodes if e.outcome == EpisodeOutcome.SUCCESS]
        if successful:
            few_shot = [
                {
                    "input": e.input_data if e.input_data else e.cue,
                    "output": e.content,
                }
                for e in successful[:2]
            ]

        if procedure:
            # Get episodic nodes for corrective hints
            episode_nodes = []
            for ep in episodes:
                node = await self._graph.get_node(f"{ep.id}_outcome")
                if node:
                    episode_nodes.append(node)

            output_schema = await self._procedural.build_schema(
                procedure, episode_context=episode_nodes
            )
            prompt_fragment = procedure.system_prompt_fragment

        # 2. Domain knowledge: retrieve via adapter
        retrieval_hints = None
        if procedure and procedure.metadata:
            retrieval_hints = procedure.metadata.get("retrieval_hints")

        domain_context = await self._domain.retrieve(
            task_description=task_description,
            task_type=proc_task_type,
            metadata=input_data,
            retrieval_hints=retrieval_hints,
        )

        has_context = bool(
            domain_context or episodes or procedure
        )
        if has_context:
            self._retrieval_hits += 1

        self._last_call_had_warnings = bool(warnings)
        if warnings:
            self._warned_calls += 1

        # Estimate token usage
        estimated_tokens = self._estimate_tokens(
            domain_context, episodes, output_schema, prompt_fragment
        )

        return PreparedContext(
            procedure=procedure,
            output_schema=output_schema,
            system_prompt_fragment=prompt_fragment,
            domain_context=domain_context,
            relevant_episodes=episodes,
            warnings=warnings,
            few_shot_examples=few_shot,
            estimated_tokens=estimated_tokens,
            capacity_used=(
                estimated_tokens / self._capacity.max_context_tokens
                if self._capacity.max_context_tokens > 0
                else 0.0
            ),
        )

    async def record_outcome(
        self,
        task_description: str,
        input_data: dict[str, Any] | None,
        output: dict[str, Any],
        outcome: EpisodeOutcome,
        correction: str | None = None,
        task_type: str | None = None,
        tags: list[str] | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> str:
        """Record an episode after an LLM call.

        1. Create episode from the call's input/output/outcome
        2. Store in episodic memory
        3. Run event-driven consolidation check
        4. Update utility scores for accessed nodes
        """
        episode = Episode(
            cue=task_description,
            content=output,
            outcome=outcome,
            correction=correction,
            task_type=task_type,
            input_data=input_data,
            tags=tags or [],
            tool_calls=tool_calls or [],
        )

        episode_id = await self._episodic.record(episode)

        # Track warning effectiveness (per-call, not cumulative)
        if outcome == EpisodeOutcome.SUCCESS and self._last_call_had_warnings:
            self._warnings_followed_by_success += 1

        # Event-driven consolidation check
        event = await self._consolidation.on_episode_recorded(episode)
        if event:
            self._consolidation.queue_event(event)

        return episode_id

    async def process_conversation(
        self,
        messages: list[dict[str, Any]],
        outcome: EpisodeOutcome,
        task_type: str | None = None,
        correction: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Auto-extract facts and record an episode from a conversation.

        Extracts reusable facts into semantic memory (with contradiction
        detection) and records the conversation as an episodic experience.

        Task type inference chain:
        1. Explicit task_type parameter (highest priority)
        2. Tool calls + params matched against registered procedures
        3. Procedure matching on conversation text
        4. None
        """
        # Parse tool calls from messages
        tool_calls = _extract_tool_calls(messages)

        # Infer task type if not provided
        if task_type is None and tool_calls:
            task_type = await self._infer_task_type_from_tools(tool_calls)
        if task_type is None:
            # Fall back to procedure matching on first user message
            first_user = next(
                (m["content"] for m in messages if m.get("role") == "user"),
                None,
            )
            if first_user:
                proc = await self._procedural.match(first_user)
                if proc:
                    task_type = proc.task_type

        # Build conversation text for LLM extraction
        conv_text = _format_conversation(messages)

        # LLM extracts facts and summary
        raw = await self._llm.generate(
            _EXTRACT_PROMPT.format(conversation=conv_text[:8000])
        )
        parsed = safe_parse_json(raw)

        # Store extracted facts in semantic memory
        if parsed and parsed.get("facts"):
            for fact in parsed["facts"]:
                if fact and isinstance(fact, str) and len(fact.strip()) > 10:
                    await self._domain.absorb(
                        insights=[fact.strip()],
                        source_episodes=[],
                        metadata={"source": "auto_extraction", "task_type": task_type},
                    )

        # Build episode from extraction
        summary = parsed.get("summary", conv_text[:500]) if parsed else conv_text[:500]
        input_summary = parsed.get("input_summary", "") if parsed else ""
        cue = input_summary or next(
            (m["content"] for m in messages if m.get("role") == "user"),
            summary,
        )

        # Collect assistant output
        assistant_content = {
            "response": " ".join(
                m["content"] for m in messages
                if m.get("role") == "assistant" and m.get("content")
            )[:2000],
        }
        if parsed and parsed.get("summary"):
            assistant_content["summary"] = parsed["summary"]

        episode = Episode(
            cue=cue[:1000],
            content=assistant_content,
            outcome=outcome,
            correction=correction,
            task_type=task_type,
            input_data={"messages_count": len(messages)},
            tags=tags or [],
            tool_calls=tool_calls,
        )

        episode_id = await self._episodic.record(episode)

        # Event-driven consolidation check
        event = await self._consolidation.on_episode_recorded(episode)
        if event:
            self._consolidation.queue_event(event)

        return episode_id

    async def _infer_task_type_from_tools(
        self, tool_calls: list[ToolCall]
    ) -> str | None:
        """Match tool call patterns against registered procedures."""
        if not tool_calls:
            return None

        # Build a signature from tool server/name/param-keys
        tool_sig = " ".join(
            f"{tc.server}/{tc.tool_name}({','.join(sorted(tc.parameters.keys()))})"
            for tc in tool_calls
        )

        # Try matching against registered procedures
        proc = await self._procedural.match(tool_sig)
        if proc:
            return proc.task_type

        # Try matching on just server/tool names
        tool_desc = " ".join(f"{tc.server} {tc.tool_name}" for tc in tool_calls)
        proc = await self._procedural.match(tool_desc)
        if proc:
            return proc.task_type

        return None

    # --- Direct subsystem access ---

    @property
    def domain(self) -> SemanticDomain:
        return self._domain

    @property
    def procedural(self) -> ProceduralMemory:
        return self._procedural

    @property
    def episodic(self) -> EpisodicMemory:
        return self._episodic

    @property
    def graph(self) -> GraphStore:
        return self._graph

    # --- Lifecycle ---

    async def consolidate(self) -> list[ConsolidationResult]:
        """Run consolidation: process queued events + periodic pass."""
        # Snapshot queued patterns BEFORE draining so periodic can skip them
        self._consolidation.snapshot_queued_patterns()
        results = await self._consolidation.process_queue()
        periodic_events = await self._consolidation.run_periodic()
        for event in periodic_events:
            result = await self._consolidation.process(event)
            if result:
                results.append(result)
        self._consolidation.clear_pattern_snapshot()
        return results

    async def stats(self) -> MemoryStats:
        return MemoryStats(
            semantic_nodes=await self._graph.node_count(MemoryType.SEMANTIC),
            procedural_nodes=await self._graph.node_count(MemoryType.PROCEDURAL),
            episodic_nodes=await self._graph.node_count(MemoryType.EPISODIC),
            total_edges=await self._graph.edge_count(),
            total_episodes=await self._episodic.episode_count(),
            pending_consolidation=len(self._consolidation._queue),
        )

    async def evaluate(self) -> EvaluationReport:
        """Generate evaluation report from episodic outcome data."""
        # Outcome trends by task type
        outcome_trend: dict[str, list[float]] = {}
        task_episodes: dict[str, list[Episode]] = {}

        all_episodes = await self._episodic.all_episodes()
        for ep in all_episodes:
            key = ep.task_type or "unknown"
            task_episodes.setdefault(key, []).append(ep)

        for task_type, eps in task_episodes.items():
            eps_sorted = sorted(eps, key=lambda e: e.timestamp)
            # Calculate rolling success rate in windows of 5
            window = 5
            rates = []
            for i in range(0, len(eps_sorted), window):
                chunk = eps_sorted[i : i + window]
                successes = sum(
                    1 for e in chunk if e.outcome == EpisodeOutcome.SUCCESS
                )
                rates.append(successes / len(chunk))
            outcome_trend[task_type] = rates

        # Retrieval hit rate
        hit_rate = (
            self._retrieval_hits / self._retrieval_attempts
            if self._retrieval_attempts > 0
            else 0.0
        )

        # Warning effectiveness (ratio of warned calls followed by success)
        warning_eff = (
            self._warnings_followed_by_success / self._warned_calls
            if self._warned_calls > 0
            else 0.0
        )

        # Hot spots: task types with low recent success rate (problem areas)
        # Cold spots: task types with too few episodes to form patterns
        cold_spots = []
        hot_spots = []
        for task_type, eps in task_episodes.items():
            if len(eps) <= 2:
                cold_spots.append(task_type)
        for task_type, rates in outcome_trend.items():
            if rates and rates[-1] < 0.5:
                hot_spots.append(task_type)

        return EvaluationReport(
            outcome_trend=outcome_trend,
            retrieval_hit_rate=hit_rate,
            warning_effectiveness=warning_eff,
            cold_spots=cold_spots,
            hot_spots=hot_spots,
        )

    # --- Private ---

    def _estimate_tokens(
        self,
        domain_context: list[DomainResult],
        episodes: list[Episode],
        output_schema: dict[str, Any] | None,
        prompt_fragment: str | None,
    ) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        total_chars = 0
        for dr in domain_context:
            total_chars += len(dr.content)
        for ep in episodes:
            total_chars += len(ep.cue) + len(str(ep.content))
        if output_schema:
            import json
            total_chars += len(json.dumps(output_schema))
        if prompt_fragment:
            total_chars += len(prompt_fragment)
        return total_chars // 4


def _extract_tool_calls(messages: list[dict[str, Any]]) -> list[ToolCall]:
    """Parse ToolCall objects from conversation messages."""
    tool_calls = []
    for msg in messages:
        for tc in msg.get("tool_calls", []):
            tool_calls.append(
                ToolCall(
                    query=tc.get("query", ""),
                    server=tc.get("server", ""),
                    tool_name=tc.get("tool_name", tc.get("name", "")),
                    parameters=tc.get("parameters", tc.get("arguments", {})),
                    result_summary=tc.get("result_summary", ""),
                    success=tc.get("success", True),
                )
            )
    return tool_calls


def _format_conversation(messages: list[dict[str, Any]]) -> str:
    """Format messages into a readable conversation string for LLM extraction."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content:
            lines.append(f"{role}: {content}")
        for tc in msg.get("tool_calls", []):
            name = tc.get("tool_name", tc.get("name", "tool"))
            params = tc.get("parameters", tc.get("arguments", {}))
            lines.append(f"  [tool call: {name}({json.dumps(params, default=str)[:200]})]")
    return "\n".join(lines)
