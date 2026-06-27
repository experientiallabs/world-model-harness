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
wmh bench run tau-bench                     # score a prompt against a committed benchmark (mean ± std)
wmh bench                                   # leaderboard across all persisted benchmark runs
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

## Benchmark results

**Open-loop reconstruction fidelity** — how faithfully the world model reproduces the *real*
recorded observation for each held-out `(state, action)`, scored 0–1 by a reference-grounded
5-dimension LLM judge (format / factuality / consistency / realism / quality), with a
deterministic-vs-volatile content split. Run with `wmh eval` (teacher-forced replay; the model never
sees the recorded observation it's scored against). Backend: Bedrock **Opus 4.8**, top-k=5 retrieval,
70/30 train-holdout, seed 0.

Comparing the un-evolved base prompt, a prompt GEPA-optimized on **tau2 only**, and one optimized
**across both benchmarks** (`world-models/cross-task/`):

| Benchmark | held-out steps | Base | tau2-only GEPA | **Cross-task GEPA** |
|---|---|---|---|---|
| **tau2-bench** (airline API) | 18 | 0.57 ± 0.36 | **0.81 ± 0.21** | 0.65 ± 0.36 |
| **terminal-tasks** (bash) | 48 | 0.60 ± 0.28 | 0.52 ± 0.28 | **0.60 ± 0.28** |
| Overall (step-weighted) | 66 | 0.589 | 0.599 | **0.617** |

Per-dimension, tau2-only optimized prompt:

| Benchmark | format | factuality | consistency | realism | quality | error-flag acc |
|---|---|---|---|---|---|---|
| tau2-bench | 1.00 | 0.62 | 0.83 | 0.96 | 0.66 | 1.00 |
| terminal-tasks | 0.66 | 0.30 | 0.57 | 0.73 | 0.34 | 0.81 |

**Reading these:** the model reproduces response *shape* and success/error status very well
(tau2 format 1.00, error-flag 1.00); the ceiling is **factuality** — predicting concrete values the
environment alone knows (a reservation's exact flights, a command's runtime output). The
**tau2-only** prompt lifts tau2 by +0.25 but *hurts* terminal (it over-confidently predicts success
on shell commands that actually fail) — a single benchmark's GEPA prompt overfits. Optimizing GEPA
**across both** benchmark families recovers terminal to baseline while keeping most of the tau2 gain,
and wins on the overall step-weighted mean — so cross-task optimization is the better general policy.

> Numbers are directional on small held-out sets, and use the offline lexical embedder (semantic
> retrieval untested). The largest factuality lever is **state grounding** — see the design note
> below.

### Design note: the world model's internal database

Today's open-loop benchmark scores the model with an **empty `state_before`** — the trace-capture
pipeline omits the environment's database to avoid leaking answers — so factuality on
records/computed values has a hard ceiling the model can't beat from `(action, retrieved demos)`
alone. This is a measurement/seeding gap, not a fundamental limit: `EnvState` already carries
`structured` (a machine-readable state dict) and `scratchpad` (the env's free-text memory, which
`WorldModel.step` already updates from each prediction's `state_note`). The direction is to **seed a
world model with its benchmark's initial database** as context and let it **read/write its own
state and memories** as a session advances — turning factuality from "guess the hidden value" into
"look it up in the state you were given."

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
