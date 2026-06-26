# Runbook: building a world model from tau2 traces (real Bedrock)

This walks the full pipeline end-to-end on real data: ingest agent traces, evolve the env prompt
with GEPA on a live LLM, persist a `.wmh/` artifact, then load it and step against it. It was run
and verified on 2026-06-25 with **Bedrock Opus 4.8** for generation and the offline
`HashingEmbedder` for retrieval (no embedding credentials required).

## 0. Source data

The traces come from `self-improvement-bench`'s tau2 benchmark — saved agent transcripts at
`self-improvement-bench/results/baseline_cache/tau2-bench/train/traces/*.json`. They are
OpenInference-shaped message logs (`system`/`user`/`assistant`) where each agent turn is a
` ```sib_bash``` ` command and the environment reply wraps `<returncode>`/`<output>`. The point is
to reuse data we already have, not to couple to SIB.

## 1. Convert SIB transcripts → OTel GenAI JSONL

The merged `otel-genai` ingestion adapter parses OTel `gen_ai.*` spans, not OpenInference message
logs, so we transform first:

```bash
uv run python scripts/sib_to_otel.py \
  /path/to/self-improvement-bench/results/baseline_cache/tau2-bench/train/traces \
  /tmp/tau2_otel.jsonl
```

Each assistant bash turn becomes an LLM span (a `bash` tool call,
`arguments={"command": ...}`); each following env reply becomes an `execute_tool` span carrying the
output and an error status when the return code is non-zero. The first customer request becomes the
trace `gen_ai.prompt` (tau).

## 2. Build the world model (ingest → split → index → GEPA → persist)

```bash
AWS_REGION=us-east-1 uv run wmh build \
  --name tau2-airline \
  --file /tmp/tau2_otel.jsonl \
  --root /tmp/tau2_wmh \
  --provider bedrock --model us.anthropic.claude-opus-4-8 --region us-east-1 \
  --gepa-budget 6
```

The build renders a guided, animated pipeline (ingest → split → index → a live GEPA rollout
progress bar with the running held-out score → a summary panel). When piped to a file it degrades to
one plain line per stage, so captured logs stay legible.

Observed (budget 6): `held_out_accuracy=0.562, frontier=2, rollouts=14`. GEPA improved the held-out
judge score from the base prompt's ~0.40 to **0.562** within the budget. The evolved prompt is
genuinely specialized — it inferred the environment is a Unix shell/tool sandbox and even captured
the exact JSON schemas the tau2 tools emit (e.g. the `get_user` record shape and key ordering).

`wmh list --root /tmp/tau2_wmh` shows every model built under the project dir.

### Artifact layout (`/tmp/tau2_wmh`)

World models are named and stored under `models/<name>/`; each is a self-contained artifact:

```
models/tau2-airline/
  config.toml              # HarnessConfig (serve provider, embed_dim, top_k, ...)
  prompts/base.txt         # the un-evolved BASE_ENV_PROMPT
  prompts/optimized.txt    # GEPA winner (what serve uses)
  prompts/frontier.json    # Pareto frontier of candidate prompts
  metrics.json             # held_out_accuracy, rollouts_used
  index/embeddings.npy     # phi(s,a) matrix for the replay buffer
  index/steps.jsonl        # the parallel Steps
```

## 3. Load the stored model and step against it

```python
from wmh.engine.world_model import WorldModel
from wmh.core.types import Action, ActionKind
from wmh.providers import get_provider, ProviderConfig, ProviderKind

provider = get_provider(ProviderConfig(
    kind=ProviderKind.BEDROCK, model="us.anthropic.claude-opus-4-8", region="us-east-1"))
wm = WorldModel.load("/tmp/tau2_wmh", provider)

s = wm.new_session(task="Customer request: I am Katherine Johnson (u_kath). Look up my account.")
print(wm.step(s.id, Action(kind=ActionKind.TOOL_CALL, name="bash",
                           arguments={"command": "get_user u_kath"})))
```

Observed:
- `get_user u_kath` → `{"membership": "silver", "name": "Katherine Johnson", "reservations": []}`,
  `is_error=False` — matches the training trace's user record (retrieval grounded the prediction).
- `get_reservation r_999` → `Error: reservation r_999 not found`, `is_error=True` — the model
  *simulates* environment behavior (errors on a missing id) rather than echoing a demo.

## 4. Play it yourself (interactive REPL)

Step into the reconstructed environment as the agent — type tool calls, the world model answers,
and the session scratchpad evolves so later turns stay consistent:

```bash
AWS_REGION=us-east-1 uv run wmh play --root /tmp/tau2_wmh --name tau2-airline \
  --task "Look up user u_kath."
# agent> get_user {"command": "get_user u_kath"}
#   -> observation panel with the env's JSON reply
# agent> :state     # show the task, turn count, and scratchpad "database"
# agent> :quit
```

Verified on 2026-06-25 against real Bedrock Opus 4.8: typing `bash {"command": "get_user u_kath"}`
returned the silver-membership user record, and a follow-up missing-id lookup returned a simulated
`not found` error — the scratchpad accumulated the state notes across turns.

## 5. Serve it over HTTP (same code path)

```bash
AWS_REGION=us-east-1 uv run wmh serve --root /tmp/tau2_wmh
# GET  /world_models                                 ->  {"world_models": ["tau2-airline"]}
# POST /world_models/tau2-airline/sessions           ->  {"session_id": ...}
# POST /world_models/tau2-airline/sessions/{id}/step  with {"action": {"kind": "tool_call", ...}}
```

`wmh serve` serves every built model by default; pass `--name` (repeatable) to serve a subset.

## Reproducing the verification automatically

`wmh/engine/integration_test.py::test_build_load_step_against_real_bedrock` runs build→load→step
against real Bedrock with a tiny budget. It is **skipped unless `AWS_REGION` is set** (same gate as
the provider live smoke tests), so the default `uv run pytest` stays offline and deterministic.

```bash
AWS_REGION=us-east-1 uv run pytest wmh/engine/integration_test.py -q   # ~37s, real LLM
```

## Notes / limitations

- The cache only had 3 tau2 transcripts (→ 3 traces, 4 steps), so the train/held-out split and the
  replay buffer are small; numbers are a smoke signal, not a benchmark.
- `embed_dim` is persisted in `config.toml` and `WorldModel.load` rebuilds the matching embedder; a
  mismatch raises a clear error instead of a cryptic numpy matmul failure.
- Embeddings stay offline by design (`HashingEmbedder`). Wiring Bedrock Titan / a real embed model
  into `BedrockProvider.embed` is a separate, additive change.
```
