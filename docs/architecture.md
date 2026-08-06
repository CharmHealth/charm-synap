# Architecture & Concepts

## The Problem

LLM agents typically treat memory as a retrieval problem: find relevant things, put them in the prompt. This fails for two reasons:

1. Attention is zero-sum. Every token in context competes for the model's attention budget, so more context produces worse per-token attention and worse reasoning. That is how transformer architectures work.

2. Retrieval only shapes the input. RAG controls what goes into the prompt but cannot shape what comes out of it. Pattern matching in the weights overrides instructions in the context, because weight activations are cheap while sustained attention on distant tokens is expensive.

Synap addresses both. Graph traversal returns specific nodes rather than text blobs. Output schemas, rather than instructions in the prompt, dictate the order in which the model produces fields.

## Three Memory Subsystems

### Semantic Memory — What the agent knows

Domain knowledge, pluggable via the `SemanticDomain` protocol.

Every project brings its own knowledge types — case law and statutes for legal research, clinical policies for healthcare, code patterns for dev tools. Synap doesn't prescribe the shape of domain knowledge. Instead, it defines a protocol with two methods:

- **`retrieve(task_description, ...)`** — return domain knowledge relevant to a task
- **`absorb(insights, source_episodes, ...)`** — store consolidated insights in your domain's schema

The built-in `SemanticMemory` implements this protocol as a generic knowledge graph with embedding-based entry points and graph traversal:

```
[lumbar fusion] --requires--> [step therapy]
[step therapy] --includes--> [physical therapy 6 weeks]
[physical therapy 6 weeks] --precedes--> [surgical authorization]
```

Retrieval happens by graph traversal from entry points rather than flat similarity search. A query about "lumbar fusion requirements" traverses `requires` and `includes` edges and returns a connected subgraph. Unrelated facts are excluded by topology, even when their embeddings are similar.

Token cost scales with graph connectivity rather than corpus size. RAG returns the top-K similar chunks; graph traversal returns a subgraph in which relationships are explicit.

Projects with domain-specific types replace `SemanticMemory` with a custom adapter that implements `SemanticDomain`. The adapter works with its own types internally and serializes to `DomainResult` at the boundary — text content, relevance score, source ID, and opaque metadata.

### Procedural Memory — How the agent reasons

A registry of task types mapped to output schemas that enforce reasoning order.

Procedural memory does not inject instructions into the prompt. It produces an output schema in which field order is the procedure. The model has to generate intermediate reasoning fields before conclusions, and each generated field conditions the next through recency bias in attention.

```python
Procedure(
    task_type="prior_auth_determination",
    field_ordering=[
        "patient_diagnosis",           # Extract diagnosis first
        "policy_criteria",             # What the policy requires
        "clinical_evidence_met",       # Does evidence satisfy criteria?
        "medical_necessity_reasoning", # Explicit reasoning chain
        "determination",               # ONLY THEN decide
    ],
    prerequisite_fields={
        "determination": ["clinical_evidence_met", "medical_necessity_reasoning"],
    },
)
```

Without this schema, the model jumps straight to `determination` based on pattern matching in the weights ("expensive procedure + chronic condition" → deny). With the schema, it must generate `clinical_evidence_met` first, which becomes high-attention context that conditions the determination.

### Episodic Memory — What the agent has experienced

A cue-tag-content graph of past agent experiences with outcomes.

Each episode is a small subgraph:
- **Cue node** — what triggered the episode (task description, input)
- **Content node** — what happened (agent output, intermediate steps)
- **Outcome node** — result (success, failure, corrected)

Retrieval is reconstructive: the system traverses from a cue through content to outcomes, building a narrative. Failed episodes are prioritized during retrieval because they are more informative for future calls.

```python
# When the agent is about to make a similar call:
ctx = await memory.prepare_call("Determine prior auth for lumbar fusion")
# ctx.warnings = ["Previous failure: missed step therapy documentation requirement"]
# ctx.few_shot_examples = [successful prior auth for similar case]
```

Episodes can also include structured records of tool invocations: which MCP server, tool, parameters, result, and success status. This data is serialized into the content node (so the LLM sees it during consolidation) and used for pattern detection. When `find_patterns` detects a tool failing repeatedly across episodes, it surfaces a tool-specific failure pattern that triggers procedural amendments (for example, "tool X requires parameter Y" or "use tool Z instead of X for this task type").

## Budget Allocation

There is no explicit budget allocator dividing tokens across subsystems. Each subsystem's data structure produces selective retrieval on its own:

| Subsystem | Structure | Why it's selective |
|---|---|---|
| Semantic | Domain adapter (graph, custom store, etc.) | Adapter returns only relevant domain knowledge |
| Procedural | Schema registry | Output schema reshapes the call, near-zero prompt tokens |
| Episodic | Cue-tag-content graph | Reconstructive retrieval returns specific episodes rather than log dumps |

The token budget stays small because each subsystem returns traversal results rather than documents.

## Consolidation

Memory evolves through consolidation: episodic experiences become domain knowledge, repeated patterns become procedural amendments, stale memories decay and evict.

Consolidation writes go through the domain adapter's `absorb()` method, so the domain decides the shape of stored knowledge rather than the consolidation engine.

The triggers are:

| Trigger | When | Example |
|---|---|---|
| Event-driven | On pattern detection | 3 failures on same task type → procedural amendment |
| Periodic | Scheduled pass | Scan episodic store for clusters → extract domain facts |
| Query-triggered | During retrieval | 5 similar episodes found → consolidate into one fact |

All triggers produce a `ConsolidationEvent`. The consolidation engine processes these asynchronously using the LLM to generate clean summaries and fact extractions, then calls `domain.absorb()` to store the result.

### Lifecycle

Every node has a `utility_score` that combines recency and access frequency, with recency gating frequency:

```
utility = (1 - decay_rate)^hours_since_last_access * (1 + frequency_bonus)
frequency_bonus = min(1, access_count / 20)
```

Recency *multiplies*, so a node that goes cold decays toward zero no matter how often it was accessed — frequency stretches the forgetting horizon (up to 2x) but never grants immortality. `hours_since_last_access` is measured from `last_accessed`, not creation, so accessing a node refreshes it. Stale nodes fade and eventually get evicted.

### Versioning

When consolidation amends a procedure, the old version stays in the graph with a `supersedes` edge:

```
[procedure_v2] --supersedes--> [procedure_v1]
```

Retirement is recorded on the node itself — a `superseded` flag in the old version's metadata — and status is read from that flag, not from the edge. The edge is kept for lineage only. Reading the flag means deleting the newer version does not resurrect the older one.

Registration reads the current active version from the graph rather than from an in-process cache, so a fresh instance over persistent storage still retires the version it finds instead of adding a second active one.

## The Shared Graph

All three subsystems operate on a single typed property graph. Nodes are partitioned by type, but edges can cross partitions, which is what lets consolidation link episodic experiences to domain facts directly.

```
[episodic: "failed auth for knee replacement"]
    --derived_from-->
[semantic: "Aetna requires step therapy documentation"]
    --step_therapy_before-->
[semantic: "Physical therapy 6 weeks"]
```

Both implementations satisfy the `GraphStore` protocol:

- **`MemoryGraph`** — in-memory, zero dependencies, good for testing and short-lived agents
- **`PersistentGraph`** — wraps a `StorageBackend`, single source of truth on disk

The consumer creates the graph and passes it to both the domain adapter and `CognitiveMemory`:

```python
graph = MemoryGraph()
domain = SemanticMemory(graph=graph, embedding_provider=embedder)
memory = CognitiveMemory(domain=domain, embedding_provider=embedder, llm_provider=llm, graph=graph)
```

This ensures all subsystems share the same graph instance.

## Storage Backends

| Backend | Graph traversal | Vector search | Persistence | Dependencies |
|---|---|---|---|---|
| `MemoryGraph` (default) | Python BFS | Python cosine | None | None |
| `KuzuBackend` | Native Cypher variable-length paths | Native `array_cosine_similarity` | File-based | `kuzu` |
| `SQLiteBackend` | Python BFS | Python cosine | File-based | `sqlite3` (stdlib) |
| `PostgresBackend` | Recursive CTE | pgvector `<=>` cosine distance | Server-based | `asyncpg` |

Kùzu is the recommended persistent backend — embedded (no server), native graph traversal via Cypher, and native vector similarity. Install with `pip install synap[kuzu]`.

```python
from synap.backends.kuzu import KuzuBackend
from synap.persistent_graph import PersistentGraph

# In-memory (default)
graph = MemoryGraph()
domain = SemanticMemory(graph=graph, embedding_provider=embedder)
memory = CognitiveMemory(domain=domain, embedding_provider=embedder, llm_provider=llm, graph=graph)

# Persistent with Kùzu
backend = KuzuBackend("./agent_memory", embedding_dim=768)
graph = PersistentGraph(backend=backend)
domain = SemanticMemory(graph=graph, embedding_provider=embedder)
memory = CognitiveMemory(domain=domain, embedding_provider=embedder, llm_provider=llm, graph=graph)
```

## Provider Model

Synap is a library rather than a framework. It does not own the agent loop, does not call the LLM for you, and does not manage conversation history. You provide two required dependencies:

- **`EmbeddingProvider`** — embeds text for entry-point matching in the graph
- **`LLMProvider`** — generates text for consolidation and bootstrapping (never for retrieval)

```python
class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

class LLMProvider(Protocol):
    async def generate(self, prompt: str, output_schema: dict | None = None) -> str: ...
```

Both protocols are async. Storage backends stay synchronous (embedded DBs don't benefit from async); `PersistentGraph` bridges with `asyncio.to_thread`.

The library prepares context (`prepare_call`) and records outcomes (`record_outcome`). The agent loop, tool calling, and the LLM client are your responsibility.

## Domain Adapters

The `SemanticDomain` protocol decouples synap from any particular knowledge representation:

```python
class SemanticDomain(Protocol):
    async def retrieve(
        self, task_description: str, task_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[DomainResult]: ...

    async def absorb(
        self, insights: list[str], source_episodes: list[MemoryNode],
        metadata: dict[str, Any] | None = None,
        node_id: str | None = None,
    ) -> str | None: ...
```

`node_id` was added in 0.2.0. Consolidation passes it on every call, so an adapter written against the 0.1.0 three-argument signature raises `TypeError` the first time consolidation runs — and because `ConsolidationEngine.process()` converts exceptions into a failed `ConsolidationResult`, that surfaces as consolidation quietly doing nothing rather than as a crash. Accept the parameter when upgrading. When it is set, store the insight under that id, replacing an existing node with the same id (this is how repeated consolidation of one pattern yields a single fact instead of near-duplicates); when it is `None`, mint your own id.

`SemanticMemory` is the built-in generic implementation — text nodes with embeddings and graph traversal. Use it to get started, replace it when your domain needs custom types.

Custom adapters work with their own types internally (Case, Statute, clinical policies, etc.) and serialize to `DomainResult` at the boundary. The information loss from collapsing domain types to text is minimal because text is the LLM interface — what the model receives is always text.
