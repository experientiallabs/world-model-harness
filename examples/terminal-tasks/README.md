# Terminal Tasks Example

This directory is a self-contained terminal-task example. It can build or evaluate a world model
from the committed `traces.otel.jsonl` corpus, regenerate that corpus from real terminal-task agent
trajectories, and run a recorded scenario against a real shell environment.

The source data is real computer-use-agent terminal runs: an LLM agent issues `bash` tool calls and
the real command output is recorded per call, including failures such as tracebacks, HTTP redirects,
and retries. The environment being reconstructed is a Unix shell.

## Files

- `traces.otel.jsonl`: OTel trace corpus consumed by `wmh build` and `wmh eval`.
- `convert_to_wmh.py`: stdlib-only converter from terminal-task trajectories.
- `run.sh`: one-command launcher for the real terminal scenario runner.
- `run_real_scenario.py`: real-environment scenario runner for comparison or trace inspection.

Generated files stay local and ignored. The converter reads source trajectories in place and only
writes the OTel JSONL corpus.

## Harness Commands

```bash
uv run wmh build --name terminal-tasks --file examples/terminal-tasks/traces.otel.jsonl
uv run wmh eval examples/terminal-tasks/traces.otel.jsonl
uv run wmh examples run terminal-tasks -- --trace 0
```

`wmh examples run terminal-tasks -- ...` invokes this folder's `run.sh` and forwards the remaining
arguments.

## Source Data

The trajectories are JSONL, one trajectory per line, with a `tool_calls` array. Each tool call has
`name`, `arguments`, `output`, and an `isError` flag:

```json
{"task": "...", "task_category": "...", "returncode": 0,
 "tool_calls": [{"name": "bash", "arguments": {"command": "..."}, "output": "...", "isError": false}]}
```

## Convert to the Corpus

```bash
cd examples/terminal-tasks
python convert_to_wmh.py <path/to/trajectories.jsonl> \
  --out traces.otel.jsonl --benchmark terminal-tasks \
  --exclude-substr <source-specific-path-fragment>
```

`--exclude-substr` is repeatable. It drops any trajectory whose raw JSON contains the given
case-insensitive substring. This is used to omit trajectories whose captured command output happens
to reference source-specific filesystem paths. It drops the whole trajectory instead of redacting a
real observation, so every committed observation stays exactly what the environment returned.

Per trajectory, the converter produces one Step per tool call:

- `action`: the real tool call, such as `bash {"command": "..."}`.
- `observation`: the real recorded output, with `is_error` from the call's `isError` flag.
- `task`: the trajectory's task instruction, carried on the first step.
- `Trace.metadata`: `benchmark`, `task_category`, and `returncode`.

`state_before` is empty because a shell has no compact, non-leaky state snapshot. Open-loop replay
reconstructs from the action, retrieved similar steps, and teacher-forced history.

## Run One Real Scenario

Use the standard examples launcher:

```bash
uv run wmh examples run terminal-tasks -- --trace 1
uv run wmh examples run terminal-tasks -- --trace 1 --cache
```

Or run the local script directly:

```bash
cd examples/terminal-tasks
./run.sh --trace 1
```

`run.sh` calls `run_real_scenario.py`, which builds a fresh Docker image from `debian:bookworm-slim`
with real `apt-get install` steps for `curl`, `python3`, `jq`, and `ca-certificates`. The build is
streamed and counted in total time before the exact recorded `bash` commands are executed.

The runner reads `traces.otel.jsonl` and reuses the harness's train/held-out split logic inline so
`--trace N` selects the same scenario consistently. These commands may hit live public APIs, so a
rerun reflects current data and can differ from the recorded observation.

Observed for `--trace 1`, cold build: 8.7s standup plus 10 commands, 10.8s total.
