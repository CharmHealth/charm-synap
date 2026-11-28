"""PostgresBackend with create_schema=False: tables from migrations, no DDL at runtime.

Deployments whose application role can't run DDL create synap's tables in their
own migrations (``schema_sql()`` gives the DDL) and construct the backend with
``create_schema=False``. Then ``init()`` creates nothing and only checks that the
tables match: present, with every column, and the configured embedding size.

Needs a live Postgres with pgvector, like ``test_postgres.py``, and a DSN whose
role may create roles (the suite creates an application role with no DDL rights).
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("psycopg_pool")

from psycopg import errors  # noqa: E402
from psycopg.conninfo import make_conninfo  # noqa: E402

from synap.backends.postgres import PostgresBackend, schema_sql  # noqa: E402
from tests.pg_pool import open_pool  # noqa: E402

DSN = os.environ.get("SYNAP_TEST_POSTGRES_DSN")

pytestmark = pytest.mark.skipif(
    not DSN, reason="set SYNAP_TEST_POSTGRES_DSN to run the Postgres backend suite"
)


def _node(node_id: str = "n1") -> dict:
    return {
        "id": node_id,
        "node_type": "semantic",
        "content": "provisioned",
        "embedding": [0.1, 0.2, 0.3],
        "created_at": "2026-01-01T00:00:00Z",
        "last_accessed": "2026-01-01T00:00:00Z",
        "metadata": {},
    }


@pytest.fixture
async def provisioned():
    """Tables created as the migrator would, plus an app role with DML only.

    Yields (prefix, app_dsn, admin_pool).
    """
    tag = uuid.uuid4().hex[:10]
    prefix = f"prov_{tag}_"
    role = f"synap_app_{tag}"
    password = uuid.uuid4().hex
    admin = await open_pool(DSN)
    async with admin.connection() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(schema_sql(prefix, 3))
        await conn.execute(f"CREATE ROLE {role} LOGIN PASSWORD '{password}'")
        await conn.execute(f"REVOKE CREATE ON SCHEMA public FROM {role}")
        await conn.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        for table in (f"{prefix}nodes", f"{prefix}edges"):
            await conn.execute(
                f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {role}"
            )
    try:
        yield prefix, make_conninfo(DSN, user=role, password=password), admin
    finally:
        async with admin.connection() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {prefix}edges CASCADE")
            await conn.execute(f"DROP TABLE IF EXISTS {prefix}nodes CASCADE")
            await conn.execute(f"DROP ROLE IF EXISTS {role}")
        await admin.close()


async def test_an_app_role_without_ddl_rights_starts_and_writes(provisioned):
    prefix, app_dsn, _ = provisioned
    pool = await open_pool(app_dsn)
    try:
        backend = PostgresBackend(
            pool, embedding_dim=3, table_prefix=prefix, create_schema=False
        )
        await backend.init()
        await backend.save_node(_node())
        loaded = await backend.load_node("n1")
    finally:
        await pool.close()
    assert loaded is not None
    assert loaded["content"] == "provisioned"


async def test_the_same_role_cannot_run_the_ddl(provisioned):
    """Proves the role above really lacks DDL rights, so the first test means something."""
    prefix, app_dsn, _ = provisioned
    pool = await open_pool(app_dsn)
    try:
        async with pool.connection() as conn:
            with pytest.raises(errors.InsufficientPrivilege):
                await conn.execute(f"CREATE TABLE {prefix}probe (id int)")
    finally:
        await pool.close()


async def test_missing_tables_are_refused_with_their_names():
    prefix = f"absent_{uuid.uuid4().hex[:10]}_"
    pool = await open_pool(DSN)
    try:
        backend = PostgresBackend(
            pool, embedding_dim=3, table_prefix=prefix, create_schema=False
        )
        with pytest.raises(RuntimeError) as exc:
            await backend.init()
    finally:
        await pool.close()
    assert f"table {prefix}nodes does not exist" in str(exc.value)
    assert f"table {prefix}edges does not exist" in str(exc.value)


async def test_a_different_embedding_size_is_refused(provisioned):
    prefix, _, admin = provisioned
    backend = PostgresBackend(
        admin, embedding_dim=4, table_prefix=prefix, create_schema=False
    )
    with pytest.raises(RuntimeError) as exc:
        await backend.init()
    assert "embedding is vector(3)" in str(exc.value)
    assert "embedding_dim=4" in str(exc.value)


async def test_a_missing_column_is_refused_by_name(provisioned):
    prefix, _, admin = provisioned
    async with admin.connection() as conn:
        await conn.execute(f"ALTER TABLE {prefix}nodes DROP COLUMN valid_until")
    backend = PostgresBackend(
        admin, embedding_dim=3, table_prefix=prefix, create_schema=False
    )
    with pytest.raises(RuntimeError) as exc:
        await backend.init()
    assert f"{prefix}nodes has no column valid_until" in str(exc.value)


async def test_init_creates_nothing_when_create_schema_is_false():
    prefix = f"none_{uuid.uuid4().hex[:10]}_"
    pool = await open_pool(DSN)
    try:
        backend = PostgresBackend(
            pool, embedding_dim=3, table_prefix=prefix, create_schema=False
        )
        with pytest.raises(RuntimeError):
            await backend.init()
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT to_regclass(%s), to_regclass(%s)",
                (f"{prefix}nodes", f"{prefix}edges"),
            )
            row = await cur.fetchone()
    finally:
        await pool.close()
    assert row == (None, None)
