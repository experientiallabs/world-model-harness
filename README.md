# World Model Harness

> **Docker as an LLM.** Stop running your evals in sandboxes — simulate your environment without running it.

A frontier LLM (Opus 4.8 / GPT 5.5) acts as the *environment* your agent steps against,
reconstructed from your own OpenTelemetry traces. No sandbox, no live services, no flaky resets.

Inspired by **Qwen-AgentWorld** (LLM-as-environment), **GEPA** (reflective prompt evolution), and
**DreamGym** (retrieval over a trace replay buffer) — but with **zero training**: we get there with
prompt optimization on a frontier model. See [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) for how
the pieces fit (and where to plug in a new provider, adapter, or embedder).

## How it works

1. **Build** from your agent's OTel traces (file export or vendor SDK): ingest → normalize → split
   train/held-out → index a replay buffer → evolve the env prompt with GEPA against the held-out split.
2. **Serve**: agents call `WorldModel.step(action)` (in-process or via the local HTTP backend). Each
   step retrieves the most similar past `(state, action) → observation` examples and predicts the
   next observation.

## Quickstart

```bash
uv sync
wmh providers verify                       # confirm Anthropic / Bedrock / Azure OpenAI / OpenAI creds
wmh build                                  # guided creation wizard (prompts for name, traces, provider…)
wmh build --name airline --file traces.jsonl   # …or fully scriptable with flags -> .wmh/models/airline/
wmh list                                   # show every built world model
wmh eval traces.jsonl                      # score reconstruction fidelity (replay + LLM judge)
wmh serve                                  # local backend on :8000 (serves all built models)
wmh demo                                   # watch an LLM agent step against the world model
wmh play                                   # step into the environment yourself (interactive REPL)
```

`wmh build` with no flags launches a **creation wizard** on an interactive terminal; pass `--file`
(and friends) or `--no-interactive` to stay scriptable. Commands that run a model (`play`/`serve`/
`demo`) take `--name`; omit it and — if several models exist — you get an interactive **picker**.

World models are **named** and stored under `.wmh/models/<name>/`, so one project can hold several
(e.g. `airline`, `retail`). Commands that read a model take `--name`; if only one is built, `--name`
is optional.

## Use it as an API

```python
from wmh import WorldModel, Action, ActionKind
from wmh.config.store import WorldModelStore
from wmh.engine.loader import load_world_model

# Resolve a named model under the project root (.wmh/models/<name>/) and load it with the
# serve provider + embedder it was built with — no need to reconstruct the provider yourself.
model_dir = WorldModelStore(".wmh").resolve("airline")
wm, _provider = load_world_model(model_dir)

session = wm.new_session(task="check out the cart")
obs = wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="add_to_cart",
                                 arguments={"sku": "A1"}))
print(obs.content)
```

Or over HTTP (same code path), namespaced by model name: `GET /world_models` to list, then
`POST /world_models/{name}/sessions` and `POST /world_models/{name}/sessions/{id}/step`.

## Providers

One interface, four backends, verified on startup. Credentials are read from the environment:

| Provider | Model | Env vars |
|---|---|---|
| Anthropic | Opus 4.8 | `ANTHROPIC_API_KEY` |
| AWS Bedrock | Claude 4.8 | `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| Azure OpenAI | GPT 5.5 | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| OpenAI | GPT 5.5 | `OPENAI_API_KEY` |

## Development

Managed with [uv](https://docs.astral.sh/uv/); linting/formatting with
[ruff](https://docs.astral.sh/ruff/); type checking with [ty](https://github.com/astral-sh/ty).

```bash
uv sync --extra dev      # create the env + install dev tools
uv run ruff check .      # lint
uv run ruff format .     # format
uv run ty check          # type check
uv run pytest -q         # tests
```

Conventions live in [AGENTS.md](./AGENTS.md). Tests are inline next to the code they cover
(`foo.py` → `foo_test.py`), organized by domain subpackage under `wmh/`.

## Status

The full pipeline works end-to-end: ingest → split → index → GEPA optimize → persist → serve/step,
verified on real Bedrock Opus 4.8 (see [`docs/tau2_runbook.md`](./docs/tau2_runbook.md)). Vendor SDK
pulls are the main stub remaining.
