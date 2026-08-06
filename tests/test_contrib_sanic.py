"""Tests for the Sanic contrib module.

Uses Sanic's built-in ASGI test client — no real server needed.
Validates the full REST API contract: knowledge CRUD, search, connect,
prepare/record, stats, consolidation, and health.

The API contract is identical to the FastAPI contrib module.
"""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("sanic")

from sanic import Sanic

from synap.contrib.sanic import create_blueprint
from synap.facade import CognitiveMemory
from synap.semantic import SemanticMemory

# Allow multiple Sanic apps with the same name across tests
Sanic.test_mode = True


@pytest.fixture
def memory(graph, embedder, llm):
    semantic = SemanticMemory(graph=graph, embedding_provider=embedder, llm_provider=llm)
    return CognitiveMemory(
        domain=semantic,
        embedding_provider=embedder,
        llm_provider=llm,
        graph=graph,
    )


@pytest.fixture
def app(memory):
    sanic_app = Sanic(f"synap_test_{uuid.uuid4().hex[:8]}")
    bp = create_blueprint(memory)
    sanic_app.blueprint(bp, url_prefix="/api/memory")
    return sanic_app


# Sanic's test client returns (request, response) tuples.
# Helpers to keep tests readable.


async def get(app, path):
    _, resp = await app.asgi_client.get(path)
    return resp


async def post(app, path, json=None):
    _, resp = await app.asgi_client.post(path, json=json)
    return resp


# ── Health ──


async def test_health(app):
    resp = await get(app, "/api/memory/health")
    assert resp.status == 200
    assert resp.json == {"status": "ok"}


# ── Store Knowledge ──


async def test_store_knowledge(app):
    resp = await post(app, "/api/memory/knowledge", json={
        "content": "The rate of profit tends to fall over time",
        "domain": "reading",
        "source": {
            "app": "stacks",
            "document": "Socialism or Extinction",
            "author": "Ted Reese",
            "location": "page 42",
            "document_type": "pdf",
        },
        "metadata": {"role": "claim", "annotation": "Key thesis of the book"},
    })
    assert resp.status == 200
    data = resp.json
    assert "id" in data
    assert data["connections_created"] == 0


async def test_store_knowledge_with_connections(app):
    r1 = await post(app, "/api/memory/knowledge", json={
        "content": "Capital accumulation leads to declining profit rates",
        "domain": "reading",
        "source": {"app": "stacks", "document": "Capital Vol. 3"},
    })
    id1 = r1.json["id"]

    r2 = await post(app, "/api/memory/knowledge", json={
        "content": "Corporate profit margins fell 12% between 2007-2009",
        "domain": "current_affairs",
        "source": {"app": "synthesis", "document": "FRED data"},
        "connections": [id1],
    })
    assert r2.status == 200
    assert r2.json["connections_created"] == 1


async def test_store_knowledge_missing_fields(app):
    resp = await post(app, "/api/memory/knowledge", json={
        "content": "some text",
        # missing domain and source
    })
    assert resp.status == 400  # Sanic raises InvalidUsage → 400


# ── Get Knowledge ──


async def test_get_knowledge(app):
    store_resp = await post(app, "/api/memory/knowledge", json={
        "content": "Reserve army of labour depresses wages",
        "domain": "reading",
        "source": {
            "app": "stacks",
            "document": "Socialism or Extinction",
            "author": "Ted Reese",
        },
        "metadata": {"role": "mechanism"},
    })
    node_id = store_resp.json["id"]

    resp = await get(app, f"/api/memory/knowledge/{node_id}")
    assert resp.status == 200
    data = resp.json
    assert data["id"] == node_id
    assert data["content"] == "Reserve army of labour depresses wages"
    assert data["domain"] == "reading"
    assert data["source"]["app"] == "stacks"
    assert data["source"]["author"] == "Ted Reese"
    assert data["metadata"]["role"] == "mechanism"


async def test_get_knowledge_not_found(app):
    resp = await get(app, "/api/memory/knowledge/nonexistent")
    assert resp.status == 404


# ── Search ──


async def test_search_knowledge(app):
    await post(app, "/api/memory/knowledge", json={
        "content": "Organic composition of capital rises over time",
        "domain": "reading",
        "source": {"app": "stacks", "document": "Capital"},
    })
    await post(app, "/api/memory/knowledge", json={
        "content": "CRISPR enables targeted gene editing",
        "domain": "reading",
        "source": {"app": "stacks", "document": "Biology textbook"},
    })

    resp = await post(app, "/api/memory/knowledge/search", json={
        "query": "capital composition",
        "max_results": 5,
    })
    assert resp.status == 200
    data = resp.json
    assert data["query"] == "capital composition"
    assert isinstance(data["results"], list)


async def test_search_knowledge_domain_scoped(app):
    await post(app, "/api/memory/knowledge", json={
        "content": "Automation displaces living labour",
        "domain": "reading",
        "source": {"app": "stacks", "document": "Socialism or Extinction"},
    })
    await post(app, "/api/memory/knowledge", json={
        "content": "US manufacturing employment declined 30% since 2000",
        "domain": "current_affairs",
        "source": {"app": "synthesis", "document": "BLS data"},
    })

    resp = await post(app, "/api/memory/knowledge/search", json={
        "query": "automation labour",
        "domain": "reading",
    })
    assert resp.status == 200
    results = resp.json["results"]
    for r in results:
        assert r["domain"] == "reading"


# ── Connect ──


async def test_connect_knowledge(app):
    r1 = await post(app, "/api/memory/knowledge", json={
        "content": "Rate of profit tends to fall",
        "domain": "reading",
        "source": {"app": "stacks", "document": "Capital"},
    })
    r2 = await post(app, "/api/memory/knowledge", json={
        "content": "Corporate margins declining since 2014",
        "domain": "current_affairs",
        "source": {"app": "synthesis", "document": "FRED"},
    })
    id1 = r1.json["id"]
    id2 = r2.json["id"]

    resp = await post(app, f"/api/memory/knowledge/{id1}/connect", json={
        "target_id": id2,
        "relation_type": "evidence_for",
        "weight": 0.8,
    })
    assert resp.status == 200
    data = resp.json
    assert data["source_id"] == id1
    assert data["target_id"] == id2
    assert data["relation_type"] == "evidence_for"
    assert "edge_id" in data


async def test_connect_knowledge_not_found(app):
    r1 = await post(app, "/api/memory/knowledge", json={
        "content": "Some fact",
        "domain": "reading",
        "source": {"app": "stacks", "document": "test"},
    })
    id1 = r1.json["id"]

    resp = await post(app, f"/api/memory/knowledge/{id1}/connect", json={
        "target_id": "nonexistent",
        "relation_type": "related_to",
    })
    assert resp.status == 404


# ── Prepare Call ──


async def test_prepare_call(app):
    await post(app, "/api/memory/knowledge", json={
        "content": "The tendency of the rate of profit to fall is a core law",
        "domain": "reading",
        "source": {"app": "stacks", "document": "Capital Vol. 3"},
    })

    resp = await post(app, "/api/memory/prepare", json={
        "task_description": "Analyze the tendency of the rate of profit to fall",
        "task_type": "analysis",
    })
    assert resp.status == 200
    data = resp.json
    assert "domain_context" in data
    assert "warnings" in data
    assert "estimated_tokens" in data
    assert "capacity_used" in data


# ── Record Outcome ──


async def test_record_outcome(app):
    resp = await post(app, "/api/memory/record", json={
        "task_description": "Classify article about sanctions",
        "output": {"classification": "geopolitics", "confidence": 0.95},
        "outcome": "success",
        "task_type": "classification",
        "tags": ["geopolitics"],
    })
    assert resp.status == 200
    assert "episode_id" in resp.json


async def test_record_outcome_invalid(app):
    resp = await post(app, "/api/memory/record", json={
        "task_description": "test",
        "output": {},
        "outcome": "invalid_outcome",
    })
    assert resp.status == 400


# ── Stats ──


async def test_stats(app):
    resp = await get(app, "/api/memory/stats")
    assert resp.status == 200
    data = resp.json
    assert "semantic_nodes" in data
    assert "procedural_nodes" in data
    assert "episodic_nodes" in data
    assert "total_edges" in data
    assert "total_episodes" in data
    assert "pending_consolidation" in data


async def test_stats_reflect_stored_knowledge(app):
    initial = (await get(app, "/api/memory/stats")).json

    await post(app, "/api/memory/knowledge", json={
        "content": "A new fact",
        "domain": "reading",
        "source": {"app": "stacks", "document": "test"},
    })

    after = (await get(app, "/api/memory/stats")).json
    assert after["semantic_nodes"] == initial["semantic_nodes"] + 1


# ── Consolidate ──


async def test_consolidate(app):
    resp = await post(app, "/api/memory/consolidate")
    assert resp.status == 200
    data = resp.json
    assert "results_count" in data
    assert isinstance(data["details"], list)


# ── Integration: full knowledge unit lifecycle ──


async def test_knowledge_lifecycle(app):
    """Store → retrieve → connect → search: the full cross-app flow."""

    # 1. Stacks stores a highlight
    r1 = await post(app, "/api/memory/knowledge", json={
        "content": "The tendency of the rate of profit to fall is the most important law of political economy",
        "domain": "reading",
        "source": {
            "app": "stacks",
            "document": "Socialism or Extinction",
            "author": "Ted Reese",
            "location": "chapter 3, page 87",
            "document_type": "pdf",
        },
        "metadata": {
            "role": "claim",
            "annotation": "Author presents this as Marx's central insight",
        },
    })
    assert r1.status == 200
    highlight_id = r1.json["id"]

    # 2. Synthesis stores an economic data point
    r2 = await post(app, "/api/memory/knowledge", json={
        "content": "US corporate profit rate declined from 8.2% to 5.1% between 2012-2024",
        "domain": "current_affairs",
        "source": {
            "app": "synthesis",
            "document": "FRED corporate profits series",
            "document_type": "metric",
        },
        "metadata": {
            "entity_type": "evidence",
            "data_source": "FRED",
        },
    })
    assert r2.status == 200
    data_id = r2.json["id"]

    # 3. Cross-domain connection (validated by user or Synthesis)
    connect_resp = await post(
        app,
        f"/api/memory/knowledge/{highlight_id}/connect",
        json={
            "target_id": data_id,
            "relation_type": "evidence_for",
        },
    )
    assert connect_resp.status == 200

    # 4. Retrieve the highlight and verify connection
    get_resp = await get(app, f"/api/memory/knowledge/{highlight_id}")
    assert get_resp.status == 200
    unit = get_resp.json
    assert unit["content"].startswith("The tendency")
    assert unit["domain"] == "reading"
    assert unit["source"]["app"] == "stacks"
    assert unit["metadata"]["role"] == "claim"
    assert data_id in unit["connections"]

    # 5. Search across domains
    search_resp = await post(app, "/api/memory/knowledge/search", json={
        "query": "rate of profit",
    })
    assert search_resp.status == 200
    results = search_resp.json["results"]
    # Both the highlight and the data point should be findable
    assert len(results) >= 1
