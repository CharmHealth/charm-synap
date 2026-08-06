# Synap

[![CI](https://github.com/CharmHealth/charm-synap/actions/workflows/ci.yml/badge.svg)](https://github.com/CharmHealth/charm-synap/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/synap)](https://pypi.org/project/synap/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Cognitive memory architecture for LLM agents.

Synap manages three types of memory — semantic, procedural, and episodic — over a shared typed property graph. Long prompts degrade reasoning quality, so the goal is to give the agent only what it needs for the task at hand. Retrieval finds entry points by vector similarity, then traverses the graph to pull in related nodes. Procedures are encoded as output schemas rather than as instructions in the prompt.

## How is this different?

Most agent memory systems (Mem0, Letta, Zep, LangMem) are retrieval over text: store chunks, find similar chunks, splice them into the prompt. Synap is built differently:

- Procedural memory uses output schemas rather than prompt instructions. Field order is reasoning order, so the model produces evidence before a conclusion without being told to "think step by step."
- Semantic memory returns connected subgraphs rather than a flat top-K list. A query about "lumbar fusion requirements" traverses `requires` and `includes` edges from the matched node.
- When the consolidation engine sees a failure pattern repeat, it generates a new schema field and registers an amended procedure version.
- Synap is a library rather than a managed service. You own the agent loop, the LLM client, and the embedding provider. Memory operations are explicit.

## Installation

```bash
pip install synap

# With Kùzu for persistent graph storage (recommended)
pip install synap[kuzu]

# With uv
uv add synap --extra kuzu
```

## Providers

Synap needs two providers you implement — one for embeddings, one for LLM text generation. Here's a minimal example using OpenAI:

```python
import openai

class OpenAIEmbedder:
    def __init__(self, client: openai.AsyncOpenAI, model: str = "text-embedding-3-small"):
        self.client = client
        self.model = model

    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in response.data]


class OpenAILLM:
    def __init__(self, client: openai.AsyncOpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model

    async def generate(self, prompt: str, output_schema: dict | None = None) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
```

Any class matching the `EmbeddingProvider` and `LLMProvider` protocols works. No inheritance required. See [docs/architecture.md](docs/architecture.md#provider-model) for details.

To try synap without wiring a real model, the package ships offline placeholders — `HashEmbedder` and `PlaceholderLLM` in `synap.providers`. They run with no external services, but are not semantic (hash-based vectors, canned LLM output), so use real providers for meaningful results.

## Quick Start

```python
from synap import (
    CognitiveMemory, CapacityHints, Procedure, EpisodeOutcome,
    SemanticMemory, MemoryGraph,
)

# Create the graph and domain adapter
graph = MemoryGraph()
domain = SemanticMemory(graph=graph, embedding_provider=your_embedder)

# You provide the embedding and LLM providers
memory = CognitiveMemory(
    domain=domain,
    embedding_provider=your_embedder,
    llm_provider=your_llm,
    graph=graph,
    capacity=CapacityHints(max_context_tokens=8192),
)

# Register a procedure — field ordering IS the enforcement
await memory.procedural.register(Procedure(
    task_type="diagnose_bug",
    description="Diagnose a bug from error logs and code context",
    schema={
        "error_classification": {"type": "string"},
        "root_cause": {"type": "string"},
        "fix_proposal": {"type": "string"},
    },
    field_ordering=["error_classification", "root_cause", "fix_proposal"],
    prerequisite_fields={"fix_proposal": ["error_classification", "root_cause"]},
))

# Seed knowledge
await domain.store("Stripe webhook payloads vary by event type; always validate shape")

# Prepare context for an LLM call
ctx = await memory.prepare_call(
    task_description="Diagnose TypeError in payment webhook handler"
)
# ctx.output_schema → enforces: classify error → find root cause → THEN propose fix
# ctx.domain_context → relevant facts from the domain adapter
# ctx.warnings → "Last time you misdiagnosed a similar TypeError..."

# Record what happened (including tool calls if any)
from synap import ToolCall

await memory.record_outcome(
    task_description="Diagnose TypeError in payment webhook handler",
    input_data={"error": "Cannot read property 'amount' of undefined"},
    output={"error_classification": "null reference", "root_cause": "...", "fix_proposal": "..."},
    outcome=EpisodeOutcome.SUCCESS,
    task_type="diagnose_bug",
    tool_calls=[
        ToolCall(
            query="find webhook handler source",
            server="code-search",
            tool_name="search_files",
            parameters={"pattern": "handleWebhook"},
            result_summary="Found src/webhooks/stripe.ts:45",
            success=True,
        ),
    ],
)
```

## Domain Adapters

Synap's semantic layer is pluggable via the `SemanticDomain` protocol. Every project brings its own knowledge types: clinical policies for healthcare, code patterns for dev tools, schema rules for data pipelines.

```python
from synap.protocols import SemanticDomain
from synap.types import DomainResult, MemoryNode

class MyDomain:
    """Implements SemanticDomain — retrieves and absorbs domain knowledge."""

    async def retrieve(self, task_description, task_type=None, metadata=None):
        # Return domain knowledge relevant to this task
        return [DomainResult(content="...", relevance=0.9, source_id="...")]

    async def absorb(self, insights, source_episodes, metadata=None):
        # Store consolidated insights in your domain's schema
        return "domain_node_id"
```

`SemanticMemory` is the built-in generic implementation: text nodes with embeddings and graph traversal. Use it to get started, replace it when your domain needs custom types.

## Persistence

By default, the graph lives in memory. Pass a storage backend for persistence:

```python
from synap.backends.kuzu import KuzuBackend
from synap.persistent_graph import PersistentGraph

backend = KuzuBackend("./agent_memory", embedding_dim=768)
graph = PersistentGraph(backend=backend)
domain = SemanticMemory(graph=graph, embedding_provider=your_embedder)

memory = CognitiveMemory(
    domain=domain,
    embedding_provider=your_embedder,
    llm_provider=your_llm,
    graph=graph,
)
```

For multi-process deployments (web servers, worker pools), use the Postgres backend:

```bash
pip install synap[postgres]
```

```python
import asyncpg
from synap.backends.postgres import PostgresBackend
from synap.persistent_graph import PersistentGraph

pool = await asyncpg.create_pool("postgresql://localhost:5432/mydb")
backend = PostgresBackend(pool, embedding_dim=768)
await backend.init()  # Creates tables (idempotent)
graph = PersistentGraph(backend=backend)
```

| Backend | Graph traversal | Vector search | Persistence | Concurrency |
|---|---|---|---|---|
| In-memory (default) | Python BFS | Python cosine | None | Single process |
| `KuzuBackend` | Native Cypher | Native `array_cosine_similarity` | File-based | Single process |
| `SQLiteBackend` | Python BFS | Python cosine | File-based | Single process |
| `PostgresBackend` | Recursive CTE | pgvector `<=>` | Server-based | Multi-process safe |

## Documentation

- [Architecture & Concepts](docs/architecture.md) — How the three memory subsystems work and why
- [API Reference](docs/api.md) — Complete interface documentation
- [Bootstrap Guide](docs/bootstrap.md) — Cold start: seeding memory from existing data
- [Examples](docs/examples.md) — Healthcare, coding agents, data pipelines

## How It Works

**Semantic memory** is pluggable via the `SemanticDomain` protocol. The built-in `SemanticMemory` stores facts as a knowledge graph and retrieves via graph traversal. Projects with domain-specific types (cases, statutes, policies, etc.) implement the protocol directly.

**Procedural memory** maps task types to output schemas. Field order is the order the model fills them in, so intermediate reasoning lands in the response before the conclusion does.

**Episodic memory** records agent experiences as cue→content→outcome subgraphs. Failed episodes are weighted higher during retrieval. Over time, repeated patterns consolidate into domain knowledge or procedural amendments. Episodes can include tool call tracking (MCP server, tool, parameters, result), so consolidation can detect tool usage patterns like wrong tool selection or malformed parameters and generate procedural amendments.

All three live on the same typed property graph. Edges cross partitions, which is what lets consolidation link episodic experiences to domain facts without a separate join step.

## Async-First

All public APIs are async. Synap is designed for integration with async frameworks (FastAPI, Sanic, etc.):

```python
# All operations are awaitable
ctx = await memory.prepare_call("task description")
episode_id = await memory.record_outcome(...)
results = await memory.consolidate()
stats = await memory.stats()
```

Storage backends stay synchronous (embedded DBs don't benefit from async). `PersistentGraph` bridges with `asyncio.to_thread`.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

MIT — see [LICENSE](LICENSE) for details.
