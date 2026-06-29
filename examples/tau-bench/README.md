# Tau-Bench Example

This directory is a self-contained, local-only tau2-bench example. It can build or evaluate a world
model from the committed `traces.otel.jsonl` corpus, regenerate that corpus from real tau2 results,
and run a recorded scenario against Sierra's real tau2 environment.

The harness intentionally does not import `tau2`. Real tau2-bench needs Python 3.12 or 3.13 and a
heavier dependency tree. Only this example folder owns that setup; the main package only consumes
the produced OTel JSONL corpus.

## Files

- `traces.otel.jsonl`: OTel trace corpus consumed by `wmh build` and `wmh eval`.
- `convert_to_wmh.py`: converter from tau2 `results.json`.
- `run.sh`: one-command launcher for the real tau2 scenario runner.
- `run_real_scenario.py`: real-environment scenario runner for comparison or trace inspection.

Generated files stay local and ignored: `.venv/`, `tau2-bench/`, and tau2 simulation data.

## Harness Commands

```bash
uv run wmh build --name tau-bench --file examples/tau-bench/traces.otel.jsonl
uv run wmh eval examples/tau-bench/traces.otel.jsonl
uv run wmh examples run tau-bench -- --trace 0
```

`wmh examples run tau-bench -- ...` invokes this folder's `run.sh` and forwards the remaining
arguments.

## Why Capture from the Real Benchmark

The world model's job is to reconstruct the actual downstream benchmark. If the corpus came from a
reimplementation, the model would learn that approximation instead of tau2-bench. This example runs
Sierra's real benchmark, including its LLM user simulator, and records what the real environment
returned.

## Setup for New Captures

```bash
cd examples/tau-bench
git clone --depth 1 https://github.com/sierra-research/tau2-bench.git
uv venv --python 3.13 .venv
uv pip install --python .venv ./tau2-bench audioop-lts boto3
export TAU2_DATA_DIR="$PWD/tau2-bench/data"
.venv/bin/tau2 check-data
```

`audioop-lts` backfills the `audioop` module removed from Python 3.13. `boto3` supports litellm's
AWS Bedrock route.

## Run a Capture

tau2-bench runs two LLM streams per task: the agent and the user simulator. Opus 4.8 on Bedrock
rejects `temperature`, so pass empty LLM args to drop sampling parameters.

```bash
cd examples/tau-bench
export TAU2_DATA_DIR="$PWD/tau2-bench/data"
export AWS_REGION=us-east-1 AWS_REGION_NAME=us-east-1

.venv/bin/tau2 run \
  --domain airline \
  --agent-llm bedrock/us.anthropic.claude-opus-4-8 --agent-llm-args '{}' \
  --user-llm  bedrock/us.anthropic.claude-opus-4-8 --user-llm-args '{}' \
  --num-trials 1 --num-tasks 12 --max-concurrency 4 \
  --save-to airline_capture
```

The output is:

```text
examples/tau-bench/tau2-bench/data/simulations/airline_capture/results.json
```

## Convert to the Corpus

```bash
cd examples/tau-bench
TAU2_DATA_DIR="$PWD/tau2-bench/data" .venv/bin/python convert_to_wmh.py \
  tau2-bench/data/simulations/airline_capture/results.json \
  --out traces.otel.jsonl --benchmark tau2-bench
```

`convert_to_wmh.py` produces one Step per agent tool call:

- `action`: the real tool call name and arguments.
- `observation`: the real recorded tool result the agent saw, with the recorded error flag.
- `Trace.metadata`: `benchmark`, `domain`, `task_id`, the task's gold `evaluation_criteria`, and
  the achieved `reward`.

`state_before` is intentionally empty for tau2. The airline and retail databases are large, and
including them would leak answers into the reconstruction task. Open-loop replay reconstructs from
the action, retrieved similar past steps, and teacher-forced session history.

Pure conversational turns are not Steps because they have no environment observation to score.

## Run One Real Scenario

Use the standard examples launcher:

```bash
uv run wmh examples run tau-bench -- --trace 0
```

Or run the local script directly:

```bash
cd examples/tau-bench
./run.sh --trace 0
```

`run.sh` sets up the `.venv` and tau2 data if needed, imports the real tau2 package, loads the
domain JSON database, and replays the exact recorded tool calls. `run_real_scenario.py` reads
`traces.otel.jsonl`, reuses the harness's train/held-out split logic inline so trace selection is
consistent, and reads the domain from each trace's metadata.

Observed for `--trace 0`, airline: 1.74s standup plus 10 tool calls, 1.74s total. tau2's environment
is an in-memory database, so this example is less about speed and more about avoiding a mandatory
tau2 dependency in the harness package.
