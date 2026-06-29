# Agent guide — world-model-harness

A frontier LLM acts as the *environment* an agent steps against, reconstructed from the user's
OpenTelemetry traces. The reusable harness lives under `wmh/`; task-specific examples live under
`examples/`.

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
   `engine`, `serving`, `cli`). The CLI is intentionally minimal; add commands only when they expose
   reusable harness behavior.

5. **Do not reintroduce top-level benchmark or artifact surfaces.** Do not add top-level
   `benchmarks/`, `docs/`, `scripts/`, `tools/`, or `world-models/` directories. Do not commit
   benchmark definitions/results or generated model artifacts outside an example folder. Built
   models normally belong under the local `.wmh/models/` artifact root; intentional prebuilt example
   artifacts belong under `examples/<task>/world_model/`.

6. **Keep dataset-specific logic inside examples.** SWE-bench, tau-bench, terminal-task, and similar
   dataset-specific launch or conversion logic belongs under `examples/<task>/`. A standard example
   folder should be self-contained, with `traces.otel.jsonl` plus task-local helpers if needed. Launch
   task helpers through `wmh examples run <task> -- <args>`.

7. **Route reusable workflows through `wmh`.** Avoid parallel top-level scripts for harness actions.
   If a workflow is generally useful outside one example dataset, implement it in `wmh/` and expose
   it through the CLI.
