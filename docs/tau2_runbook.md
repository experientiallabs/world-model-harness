# Runbook: building a world model from tau2 traces (real Bedrock)

This walks the full pipeline end-to-end on real data: ingest agent traces, evolve the env prompt
with GEPA on a live LLM, persist a `.wmh/` artifact, then load it and step against it. It was run
and verified on 2026-06-25 with **Bedrock Opus 4.8** for generation and the offline
`HashingEmbedder` for retrieval (no embedding credentials required).

## 0. Source data

A ready-to-use OTel GenAI trace corpus ships in `examples/` (one `*.otel.jsonl` per benchmark),
including `examples/tau2-bench.otel.jsonl`. These were derived from `self-improvement-bench`'s saved
agent transcripts (OpenInference message logs where each agent turn is a ` ```sib_bash``` ` command
and the environment reply wraps `<returncode>`/`<output>`) and converted into the bare OTel `gen_ai.*`
span shape the `otel-genai` ingestion adapter reads: each agent bash turn → an LLM `bash` tool-call
span, each following env reply → an `execute_tool` span (error status on non-zero return code), the
first request → the trace `gen_ai.prompt` (tau). Only turns with a paired env reply become steps.

The one-off conversion tooling is intentionally NOT committed (it's coupled to the SIB repo layout);
the committed `examples/` corpus is the reusable artifact.

## 1. Build the world model (ingest → split → index → GEPA → persist)

```bash
AWS_REGION=us-east-1 uv run wmh build \
  --file examples/tau2-bench.otel.jsonl \
  --root /tmp/tau2_wmh \
  --provider bedrock --model us.anthropic.claude-opus-4-8 --region us-east-1 \
  --gepa-budget 6
```

Observed (budget 6): `held_out_accuracy=0.562, frontier=2, rollouts=14`. GEPA improved the held-out
judge score from the base prompt's ~0.40 to **0.562** within the budget. The evolved prompt is
genuinely specialized — it inferred the environment is a Unix shell/tool sandbox and even captured
the exact JSON schemas the tau2 tools emit (e.g. the `get_user` record shape and key ordering).

### Artifact layout (`/tmp/tau2_wmh`)

```
config.toml              # HarnessConfig (serve provider, embed_dim, top_k, ...)
prompts/base.txt         # the un-evolved BASE_ENV_PROMPT
prompts/optimized.txt    # GEPA winner (what serve uses)
prompts/frontier.json    # Pareto frontier of candidate prompts
metrics.json             # held_out_accuracy, rollouts_used
index/embeddings.npy     # phi(s,a) matrix for the replay buffer
index/steps.jsonl        # the parallel Steps
```

## 2. Load the stored model and step against it

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

## 3. Serve it over HTTP (same code path)

```bash
AWS_REGION=us-east-1 uv run wmh serve --root /tmp/tau2_wmh
# POST /sessions  ->  {"session_id": ...}
# POST /sessions/{id}/step  with {"action": {"kind": "tool_call", "name": "bash", ...}}
```

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
