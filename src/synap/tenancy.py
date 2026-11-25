"""Tenant scoping — keeping one tenant's memory out of another tenant's reads.

A tenant is an opaque string chosen by the consumer (an organization, a
practice, an account). synap attaches no meaning to it beyond equality.

Every node carries a ``tenant`` (``None`` for the untenanted namespace). A
graph is bound to exactly one namespace: ``graph.scoped("acme")`` reads and
writes only nodes whose tenant is ``"acme"``, and an unscoped graph reads and
writes only nodes whose tenant is ``None``. Neither can see the other's nodes,
so there is no view in which a read silently spans tenants. That is the
point: scoping is enforced by the graph a caller holds, not by every caller
remembering to pass a filter.

Backends are the storage layer below that policy. Their read methods take a
``tenant`` keyword with three meanings:

- a string — only nodes of that tenant
- ``None`` — only untenanted nodes
- ``ALL_TENANTS`` (the default) — no filter. Kept for backend-level
  maintenance and backwards compatibility with direct backend callers; the
  graph layer never passes it for a read that returns node content.
"""

from __future__ import annotations

from typing import Any, Final


class _AllTenants:
    """Sentinel type: "do not filter by tenant". Singleton."""

    _instance: _AllTenants | None = None

    def __new__(cls) -> _AllTenants:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "ALL_TENANTS"


ALL_TENANTS: Final = _AllTenants()

TenantFilter = str | None | _AllTenants


class TenantMismatchError(ValueError):
    """A write would place or modify a node outside the graph's tenant."""


def validate_tenant(tenant: Any) -> str:
    """Return ``tenant`` if it is a usable tenant key, else raise ValueError.

    Blank strings are refused: a tenant derived from a missing value (an empty
    header, an unset field) must fail loudly rather than become a shared
    namespace every such caller lands in.
    """
    if not isinstance(tenant, str):
        raise ValueError(f"tenant must be a non-empty string, got {type(tenant).__name__}")
    if not tenant.strip():
        raise ValueError("tenant must be a non-empty string")
    return tenant


def tenant_matches(node_tenant: str | None, tenant: TenantFilter) -> bool:
    """Whether a node with ``node_tenant`` is visible under filter ``tenant``."""
    if tenant is ALL_TENANTS:
        return True
    return node_tenant == tenant
