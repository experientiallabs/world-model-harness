# Benchmarks → traces: the real trace source

The harness reconstructs an environment from **recorded traces of past agent runs**. For those
traces to be worth anything, they must come from the **actual downstream benchmark** — if we captured
from a re-implementation, the world model would learn to imitate our approximation instead of the
real benchmark. So we run the real benchmark (including its own LLM user-simulator) and record what
its real environment actually returned.

Captured benchmarks so far, each its own `examples/<benchmark>.otel.jsonl` produced by an isolated
capture tool under `tools/<benchmark>-capture/` (see [Adding a new benchmark](#adding-a-new-benchmark)):

- **`tau2-bench`** — captured live from Sierra's real tau²-bench on Bedrock Opus 4.8 (customer-service
  tool use; per-step gold rides in metadata). See `tools/tau2-capture/`.
- **`terminal-tasks`** — real computer-use-agent runs on a Unix shell (`bash` tool calls with the real
  command output recorded per call, including failures: tracebacks, HTTP 301s, retries). Converted
  from terminal-task agent trajectories. See `tools/terminal-tasks-capture/`.

## The trace contract

Each capture produces a `wmh.core.types.Trace`. Per agent **tool call**, one `Step`:

- **`action`** — the real tool call (`name` + `arguments`).
- **`observation`** — *exactly* what the real environment returned (the recorded tool result), with
  the recorded error flag. This is the open-loop ground truth the scorer grades predictions against.
- **`state_before`** — the environment state **before** the action. Optional and benchmark-dependent:
  populated only when a benchmark's state is small and non-leaky. For tau2 it is intentionally empty
  (the env DB is huge and would leak the answer — see below). Open-loop replay feeds
  `(state_before, action)` to the world model.
- **`task`** — the originating user instruction.

And **`Trace.metadata`** carries `benchmark`, `domain`, `task_id`, the task's **`gold`** evaluation
criteria (expected actions + assertions), and the achieved `reward`. Gold rides along for the
deferred **closed-loop** eval; the **open-loop** scorer ignores it (its ground truth is the recorded
observation).

Traces are stored as one-span-per-line OTel-GenAI JSONL that `wmh.ingest.otel_genai` reads. The
per-step state and trace metadata travel as optional `wmh.state.*` / `wmh.trace.metadata` span
attributes — a strict superset of the OTel GenAI semconv, so any trace that omits them still parses.

## How tau²-bench is captured

The pipeline lives in [`tools/tau2-capture/`](../tools/tau2-capture/README.md) and is deliberately
**isolated** from `wmh`:

- It runs Sierra's real [tau²-bench](https://github.com/sierra-research/tau2-bench) (`tau2 run`),
  which drives a fixed agent and an LLM user-simulator against the real domain environment. Both LLMs
  run on Bedrock Opus 4.8.
- `wmh` **never imports `tau2`**. tau²-bench needs Python 3.12–3.13 + a heavy dependency tree; `wmh`
  stays 3.11. The capture tool runs in its own `.venv`; only the produced trace JSONL is carried back
  into the repo. (`tools/` is git-ignored except the conversion script + README, and excluded from
  the `wmh` lint/type gate.)
- `convert_to_wmh.py` turns a tau2 `results.json` into the corpus: per agent tool call, the real
  action + the authoritative recorded observation the agent saw, with gold + reward + domain in
  `Trace.metadata`. tau2's `state_before` is left empty by design — the airline/retail DB is
  megabytes per step and would leak the answer (giving the model a DB that already contains the
  reservation it's asked to look up makes the eval a lookup, not a reconstruction). Open-loop replay
  reconstructs the env from the action + retrieved similar steps + teacher-forced history.

See the tool's README for the exact setup + run + convert commands.

## Adding a new benchmark

The model is one **adapter per benchmark** — a small isolated capture tool under `tools/<benchmark>-capture/`:

1. **Run the real benchmark.** Install its real upstream package in an isolated env (its own
   `.venv`, whatever Python it needs). Run it with our fixed agent on Bedrock. Do **not** add it as a
   `wmh` dependency — `wmh` must stay importable on 3.11 without it.
2. **Convert to the trace contract.** Write a `convert_to_wmh.py` that, per recorded step, emits the
   real `action` and the real recorded `observation`, and stamps `Trace.metadata` with the benchmark
   name + gold. Populate `state_before` only if the benchmark's state is small and **non-leaky** — if
   it would contain the answer to the action being scored (as tau2's full DB does), leave it empty and
   let replay reconstruct. Never invent state.
3. **Emit OTel-GenAI JSONL** in the same shape (`gen_ai.*` spans + optional `wmh.state.*` /
   `wmh.trace.metadata`) so `wmh.ingest.otel_genai` reads it with no new adapter.
4. **Commit** `examples/<benchmark>.otel.jsonl` plus the conversion script + a capture README.
   Keep the cloned upstream, venv, and raw run output git-ignored.
5. **Gate.** `uv run ruff check .`, `uv run ty check`, and `uv run pytest -q` must be clean over the
   `wmh` package (the isolated `tools/` env is excluded).
