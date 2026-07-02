# terminal-tasks trace capture (isolated)

Converts terminal-task computer-use-agent trajectories into the world-model-harness trace corpus
(`examples/terminal-tasks/traces.otel.jsonl`). These are real agent runs on a terminal/bash
environment —
an LLM agent issues `bash` tool calls and the **real command output** is recorded per call (including
real failures: tracebacks, HTTP 301s, retries). The environment being reconstructed is a Unix shell:
predict a command's real output given the command.

Like the other task examples, this is isolated from `wmh`:

- `convert_to_wmh.py` is **stdlib-only** (no `wmh` import, no third-party deps). It reads the source
  trajectories **in place** and never copies them into the repo — only the produced OTel JSONL is
  written out.
- `examples/` is excluded from the `wmh` lint/type gate.

## Prebuilt world model

This example includes the old committed terminal-tasks world model under:

```text
examples/terminal-tasks/models/terminal-tasks/
```

Use it as a local model root:

```bash
uv run wmh list --root examples/terminal-tasks
uv run wmh demo --root examples/terminal-tasks --name terminal-tasks
uv run wmh play --root examples/terminal-tasks --name terminal-tasks
```

## Source data

The trajectories ship outside this repo as JSONL, one trajectory per line, with a `tool_calls` array;
each tool call has `name`, `arguments`, `output`, and an `isError` flag:

```json
{"task": "...", "task_category": "...", "returncode": 0,
 "tool_calls": [{"name": "bash", "arguments": {"command": "..."}, "output": "...", "isError": false}]}
```

## Convert

```bash
cd examples/terminal-tasks
python convert_to_wmh.py \
  <path/to/trajectories.jsonl> \
  --out traces.otel.jsonl --benchmark terminal-tasks \
  --exclude-substr <source-specific-path-fragment>
```

`--exclude-substr` (repeatable) drops any trajectory whose raw JSON contains the given
case-insensitive substring — used to omit trajectories whose captured command output happens to
reference source-specific filesystem paths. It drops the whole trajectory rather than redacting a
real observation, so every committed observation stays exactly what the environment returned.

Per trajectory, one Step per tool call:

- `action` — the real tool call (`bash` + `{"command": ...}`).
- `observation` — the real recorded `output`, `is_error` from the call's `isError`.
- `task` — the trajectory's task instruction (on the first step as `gen_ai.prompt`).
- `Trace.metadata` — `benchmark`, `task_category`, `returncode`.

`state_before` is empty (a shell has no compact, non-leaky state snapshot; open-loop replay
reconstructs from action + retrieved steps + teacher-forced history).

The output is OTel-GenAI span JSONL that `wmh.ingest.otel_genai` reads directly.

## Concurrency scaling law

`wmh research concurrency terminal-tasks` measures how batch wall-clock falls as more scenarios
reconstruct at once (world side fans `predict_observation` across a thread pool; `--side both` also
times the matching real container, pinned to the same scenarios by `trace_id`).

```bash
AWS_REGION=us-east-1 uv run wmh research concurrency terminal-tasks \
    --scenarios 16 --levels 1,2,4,8,16 --side both --out conc.json
uv run wmh research plot-concurrency conc.json --out conc.png   # needs the viz extra
```

Each concurrent real container is built genuinely cold from scratch (unique per-run image tag,
`--no-cache` apt install) — as if on a separate machine, not sharing one warm image.

Measured (Haiku 4.5, N=16, both sides): the world model goes **26.7s → 4.1s** (W=1 → W=16, 6.5×
speedup), while standing up 16 real containers cold takes **145.6s → 31.1s** — so the world model is
**5.5–8.6× faster than the real sandbox at every level**, and the gap *grows* with concurrency (the
real side contends harder for Docker/CPU). Committed figure: `concurrency_scaling.png`.

![Concurrency scaling law on terminal-tasks](./concurrency_scaling.png)
## Run ONE real scenario (the real-environment side of the comparison)

### One command: `run.sh`

`./run.sh [--trace N]` does it end to end — sets up the venv/deps if missing, builds the
environment from scratch, runs the recorded scenario, and streams all stdout, ending with the
total time. That whole standup is the cost the world-model side skips.
Defaults to the simplest held-out scenario; `--trace N` pins one. Details below.

## Run ONE real scenario (manual)

`run_real_scenario.py` is the real half of the scenario comparison. The world-model side runs the
same held-out scenario through the model; this runs the SAME held-out scenario for real — and to be
honest about the standup the world model skips, it **builds a fresh container from scratch**
first (a `debian:bookworm-slim` base + the real `apt-get install` of `curl`, `python3`, `jq`,
`ca-certificates`), streams that build, counts it in the total time, *then* `docker exec`s the exact
recorded `bash` commands. Compare the two end times by eye.

```bash
python run_real_scenario.py --trace 1            # cold --no-cache build (default)
python run_real_scenario.py --trace 1 --cache    # reuse the cached tools image
```

Stdlib-only (needs Docker); reads the committed `traces.otel.jsonl` and
re-implements the harness's blake2b train/holdout split inline so `--trace N` matches the world-model
side. These commands hit live public APIs, so a real re-run reflects *current* data and the output
may differ from the recorded observation (rates change, releases bump) — that is the honest real
environment. Observed (`--trace 1`, cold): build from scratch 8.7s + 10 commands, 10.8s total.
