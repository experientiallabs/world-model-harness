# GEPA optimization research

This is the **research surface** for the harness's optimization trajectory: prompt optimization
(GEPA) today, heavier training methods tomorrow. It exists to try optimization directions
*empirically* — change a knob, measure reconstruction fidelity, record the result — rather than
guessing. It is the experimental sibling of [`base_prompt_iteration.md`](./base_prompt_iteration.md)
(which hand-tunes the base prompt) and [`gepa_retrieval.md`](./gepa_retrieval.md) (which explains the
RAG-aware, leak-free evaluation every experiment here inherits).

## The harness (`wmh/research`)

Three small pieces, designed so **a new experiment is one new file**:

- **`ablation.py` — the framework.** A `Condition` is a named bundle of knob values. An `Ablation`
  (Protocol) knows its `conditions()` and how to `run(condition, seed) -> float` (one build+eval at
  one seed, returning a scalar metric, higher = better). `run_ablation(ablation, seeds)` is the
  generic driver: it sweeps every condition × seed and aggregates each condition's **mean + (popula-
  tion) std across seeds** into an `AblationReport`. Aggregation is deliberately simple — small trace
  corpora make CIs/significance tests false precision.
- **`pipeline.py` — the reusable primitives.** `optimize_prompt(...)` runs `GEPAOptimizer` at a
  chosen rollout `train_temperature` + GEPA `seed`; `score_prompt(...)` replay-scores a prompt's
  held-out fidelity at a chosen `eval_temperature`. Both *wrap the real pipeline* (same prompt
  assembly, same judge, same leak-free RAG as serving and `wmh eval`), so an experiment measures
  what the harness actually does — not a reimplementation.
- **`temperature.py` — the first experiment** (below).

Backends (provider, judge, embedder) are dependency-injected via a factory, so the **unit tests
drive everything with fakes (no network)** and the live runner drives it with Bedrock through the
exact same code path.

### Adding a new experiment

Write a class satisfying `Ablation`: a `name`, a `conditions()` list, and a `run(condition, seed)`
that builds/evaluates under those knobs and returns a scalar. Reuse `optimize_prompt` /
`score_prompt` for the build+eval. The driver, seed sweep, aggregation, and reporting come for free.
Candidate next experiments: GEPA budget vs. fidelity (diminishing returns), `top_k` retrieval depth,
reflection-minibatch size, train/holdout split ratio, judge temperature.

## The knobs this added to the core (coordinate with the eval-scorer chat)

`wmh/optimize/gepa.py` historically hardcoded `temperature=0.0` in `predict_observation`. The
research harness needs to vary it, so the change is **surgical and backward-compatible**:

- `predict_observation(..., *, temperature=DEFAULT_ROLLOUT_TEMPERATURE)` — default `0.0`, i.e.
  current deterministic, serving-faithful behavior.
- `WorldModelGEPAAdapter(..., *, temperature=...)` and `GEPAOptimizer(..., *, temperature=...,
  seed=...)` thread it through; `seed` (default `0`) exposes the GEPA engine seed that was hardcoded.

Production callers (`wmh build`, `wmh serve`, `wmh eval`) pass nothing and get the old behavior. Only
the research harness sets these. The eval-scorer chat also edits `gepa.py` (it upgrades the judge) —
these are additive keyword-only params on separate lines, so conflicts should be mechanical.

## Experiment 1 — train-vs-eval temperature

**Question.** `predict_observation` was deterministic (T=0). Should it be? Temperature can matter in
two *different* places, and the harness lets us separate them:

- **Training temperature** — the temperature of the rollouts GEPA scores candidates with while it
  evolves the prompt. Higher T = noisier fitness signal, but also more diverse observations the
  reflection LM sees.
- **Evaluation/serving temperature** — the temperature the chosen prompt is replayed/served at.
  Higher T = a less reproducible environment.

We cross them into a 2×2 grid, T ∈ {0, 1} on each axis, and run it across multiple seeds. The metric
is **held-out reconstruction fidelity** (mean judge score 0..1) — the same number `wmh eval` reports.

| | T_eval = 0 | T_eval = 1 |
|---|---|---|
| **T_train = 0** | deterministic train + deterministic serve (the historical default) | det. train, variable serve |
| **T_train = 1** | variable train, det. serve | variable train + variable serve |

### Run it

```bash
AWS_REGION=us-east-1 uv run python scripts/run_temperature_ablation.py \
  examples/tau2-bench.otel.jsonl \
  --seeds 0,1 --temps 0,1 --budget 8 --out report.json
```

The runner ingests + splits the corpus exactly as `wmh build` does, sweeps the grid across seeds on
live Bedrock (offline `HashingEmbedder` for phi), prints per-cell `mean ± std`, and writes the full
`AblationReport` JSON. Swap the corpus (e.g. `terminal-bench.otel.jsonl`, which has a real
27-train/29-holdout split) or widen the grid with `--temps 0,0.5,1` for a stronger signal.

### Result (smoke run on tau2-bench)

Run on **2026-06-26**, Bedrock Opus 4.8, `--seeds 0,1 --temps 0,1 --budget 8`. tau2-bench is a tiny
corpus (1 trace → with a 0.7 split the held-out set is empty, so the runner falls back to scoring on
all steps), so these numbers are a **smoke signal that the harness works end-to-end, not a
benchmark** — read them as "the machinery runs and produces a coherent grid," then re-run on a larger
corpus for a real conclusion.

Held-out fidelity, mean ± std across seeds {0, 1}:

| condition | T_train | T_eval | mean | std |
|---|---|---|---|---|
| **Ttrain=0/Teval=0** | 0 | 0 | **0.875** | 0.025 |
| Ttrain=0/Teval=1 | 0 | 1 | 0.850 | 0.000 |
| Ttrain=1/Teval=0 | 1 | 0 | 0.850 | 0.000 |
| Ttrain=1/Teval=1 | 1 | 1 | 0.850 | 0.000 |

How to read it: compare **columns** to isolate the *evaluation* temperature effect (same trained
prompt, different serve temperature) and **rows** to isolate the *training* effect (different prompt,
same serve temperature). The deterministic/deterministic cell edges the others (one of its two seeds
scored 0.90, the rest are flat at 0.85) — directionally consistent with "T=0 doesn't hurt and may
help," but on a single held-out step that 0.025 gap is well inside noise. The honest read is a
**non-result**: no condition is distinguishable from the others at this corpus size. The value here
is the reproducible apparatus, not the number. The committed full report is
[`world-models/tau-bench/experiments/temperature.json`](../world-models/tau-bench/experiments/temperature.json);
regenerate it with the command above.

### Takeaway

T=0 remains the default for `predict_observation` (deterministic and serving-faithful) — nothing in
the smoke run argues for changing it. The point of this first experiment is to land the *apparatus*:
a one-file-per-experiment harness, the train/eval knob separation, and seed-aggregated reporting, so
the next ablation (budget, top_k, …) is cheap to run and the temperature question can be re-answered
on a larger corpus without new plumbing.

## The canonical model

`world-models/tau-bench/` is the repo's committed, GEPA-optimized example world model (built from
`examples/tau2-bench.otel.jsonl` on Bedrock Opus 4.8). It is discovered automatically by the bundled
search path (see `WorldModelStore` and [`ARCHITECTURE.md`](./ARCHITECTURE.md)), so `wmh list`,
`wmh play --name tau-bench`, and `wmh serve` find it with no `--root`. It is the model the benchmark/
reporting chat scores and the README points at.
