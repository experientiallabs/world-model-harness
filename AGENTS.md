# Agent guide — world-model-harness

A frontier LLM acts as the *environment* an agent steps against, reconstructed from the user's
OpenTelemetry traces. Three layers: an optimized base prompt, GEPA prompt evolution on a held-out
trace split, and DreamGym-style top-k retrieval over a trace replay buffer. See `README.md` for the
value prop and `DESIGN.md` (local, gitignored) for the full design.

## Toolchain

Managed with `uv`; lint/format with `ruff`; type-check with `ty`.

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format .
uv run ty check
uv run pytest -q
```

## Rules

1. **Clean tree before every commit.** Run `uv run ruff check .` and `uv run ty check` over the
   **whole project** and fix **every** error before committing — including errors you don't think
   you introduced. A commit must never add to or leave behind lint/type errors.

2. **Tests live inline next to the code.** A module `foo.py` is tested by `foo_test.py` in the same
   directory (e.g. `wmh/engine/world_model.py` → `wmh/engine/world_model_test.py`). There is no
   top-level `tests/` directory. Pytest is configured (`python_files = ["*_test.py"]`) to discover
   these.

3. **Avoid generic types.** Do not use `Any`, bare `dict`/`object`, or untyped `**kwargs` where a
   concrete type is practical. Prefer explicit pydantic models and fields; for genuinely arbitrary
   JSON use pydantic's `JsonValue` (see `wmh/core/types.py:JsonObject`), not `Any`.

4. **Keep the structure deep and the command surface small.** Code is organized into domain
   subpackages under `wmh/` (`core`, `config`, `providers`, `ingest`, `retrieval`, `optimize`,
   `engine`, `serving`, `cli`). The CLI is intentionally minimal — add commands only when needed.
