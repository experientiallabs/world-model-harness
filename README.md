# World Model Harness

> **Docker as an LLM.** Stop running your evals in sandboxes — simulate your environment without running it.

A frontier LLM (Opus 4.8 / GPT 5.5) acts as the *environment* your agent steps against,
reconstructed from your own OpenTelemetry traces. No sandbox, no live services, no flaky resets.

Inspired by **Qwen-AgentWorld** (LLM-as-environment), **GEPA** (reflective prompt evolution), and
**DreamGym** (retrieval over a trace replay buffer) — but with **zero training**: we get there with
prompt optimization on a frontier model. See [`DESIGN.md`](./DESIGN.md) for the full design.

## How it works

1. **Ingest** your agent's OTel traces (file export or vendor SDK).
2. **Build**: normalize → split train/held-out → index a replay buffer → evolve the env prompt with
   GEPA against the held-out split.
3. **Serve**: agents call `WorldModel.step(action)` (in-process or via the local HTTP backend). Each
   step retrieves the most similar past `(state, action) → observation` examples and predicts the
   next observation.

## Quickstart

```bash
uv sync
wmh init                          # scaffold .wmh/
wmh providers verify              # confirm Anthropic / Bedrock / Azure OpenAI / OpenAI creds
wmh ingest --file traces.jsonl    # or: wmh ingest --vendor <vendor>
wmh build                         # index + GEPA optimize -> .wmh/ artifact
wmh serve                         # local backend on :8000
wmh demo                          # watch an LLM agent step against the world model
```

## Use it as an API

```python
from wmh import WorldModel, Action, ActionKind
from wmh.providers import get_provider, ProviderConfig, ProviderKind

provider = get_provider(ProviderConfig(kind=ProviderKind.ANTHROPIC, model="claude-opus-4-8"))
wm = WorldModel.load(".wmh", provider)

session = wm.new_session(task="check out the cart")
obs = wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="add_to_cart",
                                 arguments={"sku": "A1"}))
print(obs.content)
```

Or over HTTP (same code path): `POST /sessions`, then `POST /sessions/{id}/step`.

## Providers

One interface, four backends, verified on startup. Credentials are read from the environment:

| Provider | Model | Env vars |
|---|---|---|
| Anthropic | Opus 4.8 | `ANTHROPIC_API_KEY` |
| AWS Bedrock | Claude 4.8 | `AWS_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| Azure OpenAI | GPT 5.5 | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| OpenAI | GPT 5.5 | `OPENAI_API_KEY` |

## Status

This is a **skeleton**: interfaces, types, CLI, and HTTP routes are wired and importable; the heavy
internals (embedding/index, GEPA loop, judge prompts, provider request mapping, vendor pulls) are
stubbed with `NotImplementedError`. See `DESIGN.md` §9 for what ships vs. what's deferred.
