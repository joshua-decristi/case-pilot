.PHONY: sync dev phoenix seed reset-demo provision-local-auth test lint typecheck eval eval-offline

UV ?= uv
UV_RUN := $(UV) run

# Live eval knobs (explicit; no defaults that spend money)
MAX_CASES ?=
BUDGET_USD ?= 5
RESULTS ?=
EVAL_ARGS ?=

sync:
	$(UV) sync --frozen --all-extras

provision-local-auth:
	$(UV_RUN) python scripts/provision_local_auth.py --write

dev:
	$(UV_RUN) python scripts/dev.py

# Separate local Phoenix viewer (pinned uvx process; not a backend dep).
# Does not start backend/simulator. Stop with Ctrl-C.
phoenix:
	$(UV_RUN) python scripts/phoenix.py

# Non-destructive: refuses overwrite of non-empty simulator DB
seed:
	$(UV_RUN) python scripts/seed.py

# Explicit destructive reseed
reset-demo:
	$(UV_RUN) python scripts/seed.py --reset

test:
	$(UV_RUN) pytest

lint:
	$(UV_RUN) ruff check src tests scripts
	$(UV_RUN) ruff format --check src tests scripts

typecheck:
	$(UV_RUN) mypy

# Offline grade only (no paid calls). Requires RESULTS=path/to/results.json
eval-offline:
	@if [ -z "$(RESULTS)" ]; then \
		echo "error: RESULTS is required (e.g. make eval-offline RESULTS=path/to/results.json)"; \
		exit 2; \
	fi
	$(UV_RUN) python -m casepilot.evaluation --offline "$(RESULTS)" $(EVAL_ARGS)

# Live paid opt-in. Requires MAX_CASES>0; budget capped at $5 via runner.
eval:
	@if [ -z "$(MAX_CASES)" ]; then \
		echo "error: MAX_CASES is required (e.g. make eval MAX_CASES=2)"; \
		exit 2; \
	fi
	@if [ "$(MAX_CASES)" -le 0 ] 2>/dev/null; then \
		echo "error: MAX_CASES must be > 0 for live eval"; \
		exit 2; \
	fi
	@awk 'BEGIN { if ($(BUDGET_USD) > 5) { print "error: BUDGET_USD may not exceed 5"; exit 2 } }'
	$(UV_RUN) python -m casepilot.evaluation --live --max-cases $(MAX_CASES) --budget-usd $(BUDGET_USD) $(EVAL_ARGS)
