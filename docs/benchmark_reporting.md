# Benchmarks + leaderboard

A **benchmark** is a first-class, committed, reproducible object: a named set of recorded trace
files plus the eval config that scores them. `wmh bench run` scores a world-model prompt against a
benchmark, persists the result, and `wmh bench` renders a leaderboard over every persisted run.

This sits on top of the open-loop eval scorer (`wmh.engine.eval`): for each held-out step it feeds
the recorded `(state, action)` teacher-forced, has the world model predict the observation, and
scores it against the *real* recorded observation with a reference-grounded LLM judge. Because the
world model runs at temperature > 0, a step yields a distribution of scores across rollouts — we
report **mean ± std**.

## The definition ("filesystem as DB")

Each benchmark is a directory under `benchmarks/<name>/`:

```
benchmarks/
  tau-bench/
    benchmark.toml      # the versioned definition (committed)
    results/            # persisted run scorecards (generated; gitignored)
```

`benchmark.toml`:

```toml
version = "1"
description = "tau2-bench tool transitions; reconstruction fidelity (open-loop)."
traces = ["../../examples/tau2-bench.otel.jsonl"]   # resolved relative to this dir

[eval]
sample_turns = "all"   # "all" | "sampled" (Qwen-AgentWorld's 5-turn protocol)
rollouts = 1           # world-model samples per scored turn -> mean ± std
temperature = 0.0      # raise with rollouts to measure variance
seeds = [0]            # one scoring pass per seed (across-seed std = reproducibility)
train_split = 0.7
top_k = 5

[eval.judge]           # pinned grader, for reproducibility
provider = "bedrock"
model = "us.anthropic.claude-opus-4-8"
region = "us-east-1"
```

Trace paths resolve relative to the benchmark directory, so a checked-out benchmark is
self-contained. The bundled `tau-bench` points at the shared `examples/` corpus (owned by the
trace-capture pipeline) so there is one source of truth.

## Running

```bash
wmh bench list                      # every committed benchmark + its eval config
wmh bench run tau-bench             # score the bundled BASE_ENV_PROMPT
wmh bench run tau-bench --model airline   # score a built model's optimized prompt
wmh bench run tau-bench --prompt my_prompt.txt
wmh bench                           # leaderboard over all persisted runs
```

`bench run` scores once per seed (the scorer draws `rollouts` samples per step internally),
aggregates to a benchmark-level **mean ± std**, and writes a `BenchRun` JSON under
`benchmarks/<name>/results/`. Runs are comparable over time; the leaderboard shows the *latest* run
per (benchmark, prompt), ranked by fidelity.

## Aggregation

- **Within a seed**: mean ± std over the seed's rollouts (the scorer's rollout distribution).
- **Across seeds**: the benchmark `fidelity_mean` is the step-weighted mean of per-seed means;
  `fidelity_std` is the population std of the per-seed means — a seed-to-seed reproducibility signal
  distinct from the within-seed rollout std.

## How it layers

`wmh/bench/` is the benchmark domain: `definition.py` (load/discover `benchmark.toml`), `runner.py`
(seed sweep → `BenchRun`), `scoring.py` (the one binding to `wmh.engine.eval`), `results.py`
(persist/aggregate), `leaderboard.py` (rank persisted runs). The CLI commands are thin wrappers;
the rich tables live in `wmh.cli.ui`.
