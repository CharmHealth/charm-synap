"""CLI entrypoint for synap serve."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="synap", description="Synap memory service")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Start the HTTP API server")
    serve_parser.add_argument(
        "--framework",
        choices=["fastapi", "sanic"],
        default="fastapi",
        help="Web framework to use (default: fastapi)",
    )
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=8100, help="Bind port")
    serve_parser.add_argument(
        "--backend",
        choices=["memory", "sqlite", "kuzu", "postgres"],
        default="memory",
        help="Storage backend (default: memory)",
    )
    serve_parser.add_argument(
        "--db-path",
        default="./synap_data",
        help="Database path for sqlite/kuzu backends",
    )
    serve_parser.add_argument(
        "--db-url",
        help="Database URL for postgres backend",
    )
    serve_parser.add_argument(
        "--embedding-dim",
        type=int,
        default=768,
        help="Embedding dimension (default: 768)",
    )

    args = parser.parse_args()

    if args.command != "serve":
        parser.print_help()
        sys.exit(1)

    _serve(args)


def _serve(args: argparse.Namespace) -> None:
    """Start the synap HTTP server.

    Note: This requires an EmbeddingProvider and LLMProvider to be configured.
    For production use, create the CognitiveMemory instance programmatically
    and use create_app() directly. This CLI is for development/testing.
    """
    # Lazy imports so CLI is fast when just checking --help
    from synap.facade import CognitiveMemory
    from synap.semantic import SemanticMemory

    # Build graph
    graph = _build_graph(args)

    # Placeholder providers — production deployments should use create_app() directly
    from synap.contrib._providers import StubEmbedder, StubLLM

    embedder = StubEmbedder(dim=args.embedding_dim)
    llm = StubLLM()

    semantic = SemanticMemory(graph=graph, embedding_provider=embedder, llm_provider=llm)
    memory = CognitiveMemory(
        domain=semantic,
        embedding_provider=embedder,
        llm_provider=llm,
        graph=graph,
    )

    print(f"Starting synap server ({args.framework}) on {args.host}:{args.port}")
    print(f"Backend: {args.backend}")
    print("Warning: Using stub embedding/LLM providers. For production, use create_app() directly.")

    if args.framework == "fastapi":
        import uvicorn

        from synap.contrib.fastapi import create_app

        app = create_app(memory)
        uvicorn.run(app, host=args.host, port=args.port)

    elif args.framework == "sanic":
        from synap.contrib.sanic import create_app

        app = create_app(memory)
        app.run(host=args.host, port=args.port, single_process=True)


def _build_graph(args: argparse.Namespace):
    """Build the appropriate graph store from CLI args."""
    if args.backend == "memory":
        from synap.graph import MemoryGraph
        return MemoryGraph()

    elif args.backend == "sqlite":
        from synap.backends.sqlite import SQLiteBackend
        from synap.persistent_graph import PersistentGraph
        return PersistentGraph(backend=SQLiteBackend(args.db_path))

    elif args.backend == "kuzu":
        from synap.backends.kuzu import KuzuBackend
        from synap.persistent_graph import PersistentGraph
        return PersistentGraph(
            backend=KuzuBackend(args.db_path, embedding_dim=args.embedding_dim)
        )

    elif args.backend == "postgres":
        if not args.db_url:
            print("Error: --db-url required for postgres backend", file=sys.stderr)
            sys.exit(1)
        # Postgres is async — handled differently
        from synap.backends.postgres import PostgresBackend
        from synap.persistent_graph import PersistentGraph
        return PersistentGraph(backend=PostgresBackend(args.db_url))

    raise ValueError(f"Unknown backend: {args.backend}")


if __name__ == "__main__":
    main()
