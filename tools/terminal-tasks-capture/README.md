# terminal-tasks trace capture (isolated)

Converts terminal-task computer-use-agent trajectories into the world-model-harness trace corpus
(`examples/terminal-tasks.otel.jsonl`). These are real agent runs on a terminal/bash environment —
an LLM agent issues `bash` tool calls and the **real command output** is recorded per call (including
real failures: tracebacks, HTTP 301s, retries). The environment being reconstructed is a Unix shell:
predict a command's real output given the command.

Like `tools/tau2-capture/`, this is isolated from `wmh`:

- `convert_to_wmh.py` is **stdlib-only** (no `wmh` import, no third-party deps). It reads the source
  trajectories **in place** and never copies them into the repo — only the produced OTel JSONL is
  written out.
- `tools/` is excluded from the `wmh` lint/type gate.

## Source data

The trajectories ship outside this repo as JSONL, one trajectory per line, with a `tool_calls` array;
each tool call has `name`, `arguments`, `output`, and an `isError` flag:

```json
{"task": "...", "task_category": "...", "returncode": 0,
 "tool_calls": [{"name": "bash", "arguments": {"command": "..."}, "output": "...", "isError": false}]}
```

## Convert

```bash
python tools/terminal-tasks-capture/convert_to_wmh.py \
  <path/to/trajectories.jsonl> \
  --out examples/terminal-tasks.otel.jsonl --benchmark terminal-tasks \
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

## Run ONE real scenario (the real-environment side of the comparison)

`run_real_scenario.py` is the real half of the scenario comparison. The world model side is
`wmh bench scenario terminal-tasks --trace N`; this runs the SAME held-out scenario for real — and
to be honest about the standup the world model skips, it **builds a fresh container from scratch**
first (a `debian:bookworm-slim` base + the real `apt-get install` of `curl`, `python3`, `jq`,
`ca-certificates`), streams that build, counts it in the total time, *then* `docker exec`s the exact
recorded `bash` commands. Compare the two end times by eye.

```bash
python run_real_scenario.py --trace 1            # cold --no-cache build (default)
python run_real_scenario.py --trace 1 --cache    # reuse the cached tools image
```

Stdlib-only (needs Docker); reads the committed `examples/terminal-tasks.otel.jsonl` and
re-implements the harness's blake2b train/holdout split inline so `--trace N` matches the world-model
side. These commands hit live public APIs, so a real re-run reflects *current* data and the output
may differ from the recorded observation (rates change, releases bump) — that is the honest real
environment. Observed (`--trace 1`, cold): build from scratch 8.7s + 10 commands, 10.8s total.
