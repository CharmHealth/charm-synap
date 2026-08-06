# Contributing to Synap

Thanks for your interest in contributing to Synap! This guide will help you get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/CharmHealth/charm-synap.git
cd charm-synap

# Create a virtual environment and install dev dependencies
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Or with uv
uv sync --extra dev
```

## Running Tests

```bash
pytest
```

Tests use fake embedding and LLM providers (defined in `tests/conftest.py`), so no external services are needed.

## Making Changes

1. **Fork the repo** and create a branch from `main`
2. **Write tests** for any new functionality
3. **Run the test suite** to make sure nothing is broken
4. **Keep your PR focused** — one concern per pull request
5. **Update documentation** if you're changing public API

## Code Style

- Type hints on all public functions and methods
- Docstrings on public classes and methods
- Follow the existing patterns — protocols for extension points, dataclasses for data
- Async-first: public API methods should be async

## Architecture Notes

Before diving in, read [docs/architecture.md](docs/architecture.md) to understand the three-subsystem design and why decisions were made.

Key principles:
- **Protocols over inheritance** — consumers implement `EmbeddingProvider`, `LLMProvider`, `SemanticDomain`, etc.
- **Zero required dependencies** — core library has no runtime deps; backends are optional
- **Graph is the integration layer** — all subsystems share one typed property graph

## What to Work On

- Check [open issues](../../issues) for bugs and feature requests
- Issues labeled `good first issue` are a great starting point
- If you want to work on something larger, open an issue first to discuss the approach

## Pull Requests

- Fill out the PR template
- Ensure tests pass
- Keep commits focused and well-described
- Be open to feedback — code review is collaborative, not adversarial

## Questions?

Open a [discussion](../../discussions) or comment on the relevant issue.
