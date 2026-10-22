# CasePilot

Local portfolio demo package bootstrap (Python 3.14+).

## Development bootstrap

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

## Tool commands (valid now)

```bash
# Install / refresh lock-consistent env
uv sync --extra dev

# Lint
uv run ruff check src
uv run mypy

# Tests (once tests land)
uv run pytest
```

Package import path: `src/casepilot` (set via `pythonpath` in pyproject / `PYTHONPATH=src`).
