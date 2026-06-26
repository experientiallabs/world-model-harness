# tau2-bench trace capture (isolated)

This directory is a **self-contained, local-only capture tool**. It runs the *real*
[tau²-bench](https://github.com/sierra-research/tau2-bench) benchmark and converts its trajectories
into the world-model-harness trace corpus (`examples/tau2-bench.otel.jsonl`).

It is deliberately isolated:

- **`wmh` never imports `tau2`.** Real tau²-bench needs Python 3.12–3.13 and a heavy dependency tree
  (`litellm`, `boto3`, …); `wmh` stays on 3.11. This tool runs in its own `.venv`. Only the produced
  trace JSONL is carried back into the repo.
- The cloned `tau2-bench/`, the `.venv/`, and any `data/` simulations are **git-ignored**. Only
  `convert_to_wmh.py` and this README are tracked (they are our reproducible tooling).
- `tools/` is excluded from the `wmh` lint/type gate (`pyproject.toml`), since it targets a different
  Python and imports a package `wmh` doesn't depend on.

## Why capture from the REAL benchmark

The world model's job is to reconstruct the **actual downstream benchmark**. If we captured traces
from a re-implementation, the model would learn to imitate our approximation, not tau²-bench. So we
run Sierra's real benchmark — including its **LLM user-simulator** — and record what its real
environment actually returned.

## Setup

```bash
cd tools/tau2-capture
git clone --depth 1 https://github.com/sierra-research/tau2-bench.git
uv venv --python 3.13 .venv
uv pip install --python .venv ./tau2-bench audioop-lts boto3
#   audioop-lts: backport of the audioop module removed from Python 3.13 stdlib (tau2 imports it)
#   boto3:       litellm's AWS Bedrock route
export TAU2_DATA_DIR="$PWD/tau2-bench/data"
.venv/bin/tau2 check-data    # should report OK
```

## Run a capture (live, on Bedrock Opus 4.8 — the only creds available here)

tau²-bench runs two LLM streams per task (the agent and the user-simulator). Opus 4.8 on Bedrock
rejects the `temperature` parameter, so pass empty LLM args to drop it.

```bash
export TAU2_DATA_DIR="$PWD/tau2-bench/data" AWS_REGION=us-east-1 AWS_REGION_NAME=us-east-1
.venv/bin/tau2 run \
  --domain airline \
  --agent-llm bedrock/us.anthropic.claude-opus-4-8 --agent-llm-args '{}' \
  --user-llm  bedrock/us.anthropic.claude-opus-4-8 --user-llm-args '{}' \
  --num-trials 1 --num-tasks 12 --max-concurrency 4 \
  --save-to airline_capture
# -> tau2-bench/data/simulations/airline_capture/results.json
```

## Convert to the wmh corpus

```bash
TAU2_DATA_DIR="$PWD/tau2-bench/data" .venv/bin/python convert_to_wmh.py \
  tau2-bench/data/simulations/airline_capture/results.json \
  --out ../../examples/tau2-bench.otel.jsonl --benchmark tau2-bench
```

`convert_to_wmh.py` produces, per simulation, one Step per agent **tool call**:

- `action` — the real tool call (name + arguments).
- `observation` — the **real recorded tool result** the agent saw (`gen_ai.tool.message`), error flag
  from the recorded `error`.
- `Trace.metadata` — `benchmark`, `domain`, `task_id`, the task's **gold** `evaluation_criteria`
  (expected actions + assertions), and the achieved `reward`.

`state_before` is intentionally **empty** for tau2. The airline/retail DB (full flight catalog, all
reservations, all users) is megabytes per step *and* would leak the answer — handing the model a DB
that already contains reservation `NM1VX1` turns predicting `get_reservation_details(NM1VX1)` into a
lookup, not a reconstruction. Open-loop replay reconstructs the env from the action + retrieved
similar past steps + the teacher-forced session history, which is the whole point. (The wmh adapter
still *reads* `wmh.state.*` when present, for future benchmarks whose state is small and non-leaky.)

Pure-conversational turns (no tool call) are not Steps: open-loop replay scores predicted
observations for `(state, action)`, and a chat turn has no environment observation to score.

The output is OTel-GenAI span JSONL that `wmh.ingest.otel_genai` reads directly (the per-step state
and gold travel as optional `wmh.state.*` / `wmh.trace.metadata` attributes).
