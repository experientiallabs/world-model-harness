# SWE-bench trace capture (isolated)

This directory is a **self-contained, local-only capture tool**. It runs the *real*
[SWE-bench Verified](https://www.swebench.com/) benchmark with the standard
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) harness and converts the recorded
agent trajectories into the world-model-harness trace corpus (`examples/swe-bench.otel.jsonl`).

It is deliberately isolated, exactly like `tools/tau2-capture/` and `tools/terminal-tasks-capture/`:

- **`wmh` never imports SWE-bench or mini-swe-agent.** SWE-bench Verified runs each instance in its
  own per-instance Docker image (the buggy repo at a pinned commit + its full test env); the agent
  harness needs its own heavy dependency tree. This tool runs in its own `.venv` and uses the local
  Docker daemon. Only the produced trace JSONL is carried back into the repo.
- The cloned harness, the `.venv/`, the pulled Docker images, and the raw run output are
  **git-ignored**. Only `convert_to_wmh.py` and this README are tracked.
- `tools/` is excluded from the `wmh` lint/type gate (`pyproject.toml`), since it targets a different
  Python and imports packages `wmh` doesn't depend on.

## Why capture from the REAL benchmark

The world model's job is to reconstruct the **actual downstream benchmark**. SWE-bench's environment
is a real shell inside a real repo container: the agent runs commands (`ls`, `cat`, `sed`, `python
-m pytest`, …) and the environment returns the **real** stdout/stderr + exit code — including
tracebacks, build errors, and test logs. We record exactly what the real environment returned, never
a re-implementation. (This also makes SWE-bench the *low-fidelity* end of the spectrum for a world
model: its observations are arbitrary code-execution output, the hardest thing to reconstruct — by
design, so the harness is honest about where it's hard.)

## Setup

```bash
cd tools/swe-bench-capture
git clone --depth 1 https://github.com/SWE-agent/mini-swe-agent.git
uv venv --python 3.12 .venv
uv pip install --python .venv ./mini-swe-agent 'swebench' boto3
#   swebench: the Verified dataset loader + (optionally) the official evaluation harness
#   boto3:    litellm's AWS Bedrock route
# Docker must be running locally; the agent execs commands inside the per-instance images.
```

## Run a capture (live, on Bedrock Opus 4.8 — the only creds available here)

mini-swe-agent's `swebench` runner pulls each instance's Docker image, runs the agent loop inside it,
and writes one `<instance_id>.traj.json` per instance under the output dir. Opus 4.8 on Bedrock
rejects the `temperature` parameter, so the model config must not set one.

```bash
export AWS_REGION=us-east-1 AWS_REGION_NAME=us-east-1
# A SMALL slice: --subset verified, the first few instances. Each instance pulls a multi-GB
# x86_64 image (runs under emulation on arm64) and runs a real agent loop, so keep the slice small.
.venv/bin/python -m minisweagent.run.benchmarks.swebench \
  --subset verified --split test --slice 0:3 \
  --environment-class docker \
  -m bedrock/us.anthropic.claude-opus-4-8 \
  -o runs/verified_capture
# -> runs/verified_capture/<instance_id>/<instance_id>.traj.json  (one per instance)
```

(Flag names follow the installed mini-swe-agent version — see `python -m
minisweagent.run.benchmarks.swebench --help`. The shape that matters: a per-instance `*.traj.json`
whose `messages` are the recorded agent loop. The default model config must not set `temperature`
since Opus 4.8 rejects it; the bundled `swebench.yaml` config is a fine base.)

## Convert to the wmh corpus

```bash
.venv/bin/python convert_to_wmh.py \
  runs/verified_capture \
  --out ../../examples/swe-bench.otel.jsonl --benchmark swe-bench
```

`convert_to_wmh.py` (stdlib-only, no `wmh` import) reads every `*.traj.json` under the run dir and
produces, per trajectory, one Step per agent **shell command**:

- `action` — the real command the agent ran (`bash {"command": "..."}`), parsed from the assistant
  message's fenced command block.
- `observation` — the **real recorded command output** the agent saw (the following environment
  message: stdout/stderr inside `<output>…</output>`), with `is_error` from the recorded
  `<returncode>` being non-zero.
- `task` — the instance's problem statement (the GitHub issue), carried on the first step.
- `Trace.metadata` — `benchmark`, `instance_id`, `repo`, and the gold `model_patch`/`exit_status`
  when present (gold rides along for the deferred closed-loop eval; the open-loop scorer ignores it).

`state_before` is left **empty**: the environment state is an entire repo working tree (huge, and a
`cat` of the buggy file would leak the answer to "what does this file contain?"). Open-loop replay
reconstructs from the action + retrieved similar steps + teacher-forced history — the whole point.

Pure-reasoning turns (assistant messages with no command) are not Steps: open-loop replay scores a
predicted observation for `(state, action)`, and a reasoning turn has no environment observation.

The output is OTel-GenAI span JSONL that `wmh.ingest.otel_genai` reads directly.
