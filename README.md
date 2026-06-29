# World Model Harness

> **Docker as an LLM.** Simulate an agent environment from traces instead of standing up a sandbox.

A frontier LLM acts as the environment your agent steps against, reconstructed from OpenTelemetry
traces. The harness ingests recorded `(state, action) -> observation` steps, builds a retrieval index,
evolves the base environment prompt with GEPA, and serves the resulting world model locally.

## How It Works

1. **Build** from OTel traces: ingest, normalize, split train/held-out, index the replay buffer, and
   optimize the environment prompt.
2. **Serve or play** the built model: agents call `WorldModel.step(action)` in-process or through the
   local HTTP backend.
3. **Evaluate** reconstruction fidelity with `wmh eval` against trace files.

## Quickstart

```bash
uv sync
wmh providers verify
wmh build --name airline --file examples/tau-bench/traces.otel.jsonl
wmh list
wmh eval examples/tau-bench/traces.otel.jsonl
wmh examples list
wmh examples run tau-bench -- --trace 0
wmh serve
wmh demo --name airline
wmh play --name airline
```

`wmh build` with no flags launches a guided creation wizard on an interactive terminal. Pass
`--file` and related flags, or `--no-interactive`, for scriptable runs.

World models are named and stored under `.wmh/models/<name>/`. `wmh list`, `wmh serve`, `wmh demo`,
and `wmh play` only use models built locally in that directory.

## Examples

Dataset-specific logic lives only under `examples/`. Each task folder is self-contained:

- `examples/swe-bench/traces.otel.jsonl`
- `examples/tau-bench/traces.otel.jsonl`
- `examples/terminal-tasks/traces.otel.jsonl`

Each example folder may include task-local capture or launch helpers. Launch them through
`wmh examples run <task> -- <args>`. Reusable harness behavior belongs in `wmh/` and should be
exposed through the `wmh` CLI.

## Python API

```python
from wmh import Action, ActionKind
from wmh.config.store import WorldModelStore
from wmh.engine.loader import load_world_model

model_dir = WorldModelStore(".wmh").resolve("airline")
wm, _provider = load_world_model(model_dir)

session = wm.new_session(task="check out the cart")
obs = wm.step(
    session.id,
    Action(kind=ActionKind.TOOL_CALL, name="add_to_cart", arguments={"sku": "A1"}),
)
print(obs.content)
```

Over HTTP, use `GET /world_models`, then `POST /world_models/{name}/sessions` and
`POST /world_models/{name}/sessions/{id}/step`.

## Providers

Credentials are read from the environment.

| Provider | Default model family | Env vars |
|---|---|---|
| Anthropic | Claude Opus | `ANTHROPIC_API_KEY` |
| AWS Bedrock | Claude Opus | `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| Azure OpenAI | GPT | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| OpenAI | GPT | `OPENAI_API_KEY` |

## Development

```bash
uv sync --extra dev
uv run ruff check .
uv run ruff format .
uv run ty check
uv run pytest -q
```

Conventions live in `AGENTS.md`. Tests are inline next to the code they cover
(`foo.py` -> `foo_test.py`) under `wmh/`.
