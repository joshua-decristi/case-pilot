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

## Shared contracts & settings (landed)

- `casepilot.shared.contracts` — shape-only Pydantic models/enums for simulator + backend HTTP payloads and route templates.
- `casepilot.shared.config.Settings` / `get_settings()` — env-backed settings (`pydantic-settings`); service/operator tokens required, OpenRouter key optional until live runs. See `.env.example` (placeholders only).
- Local auth: `python scripts/provision_local_auth.py` reports missing tokens; `--write` appends random tokens to `.env` without printing or overwriting existing keys.

No backend, simulator, CLI, or apps in this slice yet.
