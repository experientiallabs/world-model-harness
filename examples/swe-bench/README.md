# SWE-bench Example

This directory is a self-contained, local-only SWE-bench example. It can build or evaluate a world
model from the committed `traces.otel.jsonl` corpus, regenerate that corpus from real
mini-swe-agent trajectories, and run a recorded scenario against the real SWE-bench environment.

SWE-bench Verified runs each instance in its own Docker image: the buggy repo at a pinned commit plus
its full test environment. The harness intentionally does not import SWE-bench or mini-swe-agent.
Only this example folder knows how to set up those dependencies and convert their recorded
trajectories.

## Files

- `traces.otel.jsonl`: OTel trace corpus consumed by `wmh build` and `wmh eval`.
- `convert_to_wmh.py`: stdlib-only converter from mini-swe-agent trajectories.
- `run.sh`: one-command launcher for the real SWE-bench scenario runner.
- `run_real_scenario.py`: real-environment scenario runner for comparison or trace inspection.

Generated files stay local and ignored: `.venv/`, `mini-swe-agent/`, run output, and Docker images.

## Harness Commands

```bash
uv run wmh build --name swe-bench --file examples/swe-bench/traces.otel.jsonl
uv run wmh eval examples/swe-bench/traces.otel.jsonl
uv run wmh examples run swe-bench -- --trace 0
```

`wmh examples run swe-bench -- ...` invokes this folder's `run.sh` and forwards the remaining
arguments.

## Why Capture from the Real Benchmark

The world model's job is to reconstruct the actual downstream environment. SWE-bench is a real shell
inside a real repository container: the agent runs commands such as `ls`, `cat`, `sed`, and
`python -m pytest`, and the environment returns real stdout, stderr, exit codes, tracebacks, build
errors, and test logs.

The converter records exactly what the real environment returned. It does not reimplement the
benchmark. SWE-bench is also the hardest example here for a world model because its observations are
arbitrary code execution output.

## Setup for New Captures

```bash
cd examples/swe-bench
git clone --depth 1 https://github.com/SWE-agent/mini-swe-agent.git
uv venv --python 3.12 .venv
uv pip install --python .venv ./mini-swe-agent 'swebench' boto3
```

Docker must be running locally. `swebench` provides the Verified dataset loader and evaluation
harness. `boto3` supports litellm's AWS Bedrock route.

## Run a Capture

mini-swe-agent's SWE-bench runner pulls each instance's Docker image, runs the agent loop inside it,
and writes one `<instance_id>.traj.json` per instance under the output directory.

Opus 4.8 on Bedrock rejects the `temperature` parameter, so the model config must not set one.

```bash
cd examples/swe-bench
export AWS_REGION=us-east-1 AWS_REGION_NAME=us-east-1

.venv/bin/python -m minisweagent.run.benchmarks.swebench \
  --subset verified --split test --slice 0:3 \
  --environment-class docker \
  -m bedrock/us.anthropic.claude-opus-4-8 \
  -o runs/verified_capture
```

Flag names follow the installed mini-swe-agent version. Check:

```bash
.venv/bin/python -m minisweagent.run.benchmarks.swebench --help
```

The important output shape is a per-instance `*.traj.json` whose `messages` are the recorded agent
loop.

## Convert to the Corpus

```bash
cd examples/swe-bench
.venv/bin/python convert_to_wmh.py runs/verified_capture \
  --out traces.otel.jsonl --benchmark swe-bench
```

`convert_to_wmh.py` reads every `*.traj.json` under the run directory and produces one Step per
agent shell command:

- `action`: the real command the agent ran, encoded as `bash {"command": "..."}`.
- `observation`: the real recorded command output the agent saw, with `is_error` set from the
  recorded non-zero return code.
- `task`: the instance problem statement, carried on the first step.
- `Trace.metadata`: `benchmark`, `instance_id`, `repo`, and any gold `model_patch` or `exit_status`
  present in the source trajectory.

`state_before` is left empty. The environment state is an entire repo working tree, and including it
would be huge and often leaky. Open-loop replay reconstructs from the action, retrieved similar
steps, and teacher-forced history.

Reasoning-only assistant turns are not Steps because they have no environment observation to score.

## Run One Real Scenario

Use the standard examples launcher:

```bash
uv run wmh examples run swe-bench -- --trace 0
uv run wmh examples run swe-bench -- --trace 0 --cache
```

Or run the local script directly:

```bash
cd examples/swe-bench
./run.sh --trace 0
```

`run.sh` sets up the `.venv` if needed, installs dependencies, stands up the real SWE-bench
environment, and streams stdout. By default the standup is cold: the runner purges local SWE-bench
images and builds with `--no-cache`. Pass `--warm` or `--cache` for faster repeat runs, and
`--keep-image` to keep the stood-up images.

`run_real_scenario.py` reads `traces.otel.jsonl` and reuses the harness's train/held-out split
logic inline so `--trace N` selects the same scenario consistently. Observed for
`astropy__astropy-13453`, `--trace 0`, cold build: 339.5s standup plus 19 commands, 362.0s total.
