# Trace scaling law

**Question:** as we feed the harness more recorded traces, does world-model reconstruction fidelity
keep climbing — and where does it saturate? This is the trace analogue of a data-scaling law: the
x-axis is the number of *training* traces, the y-axis is held-out fidelity.

It is a standard experiment in the GEPA research harness (`wmh.research`, see
[`gepa_research.md`](./gepa_research.md)): an `Ablation` swept across seeds by `run_ablation`, so each
point on the curve comes with an across-seed error bar. It reuses the canonical
`optimize_prompt` / `score_prompt` (the same GEPA + replay the harness ships), so the fidelity is
directly comparable to `wmh eval`.

## Design

### Fixed test, growing train (`wmh/research/scaling_split.py`)

To read a clean curve the evaluation target must not move. We carve the corpus by a stable hash of
`trace_id` into three bands:

```
corpus
 ├─ test   (fixed fraction; never seen by GEPA — the reported fidelity)
 ├─ valid  (fixed fraction; GEPA selects its best candidate on this)
 └─ train pool  (everything else; we sample N from here)
```

- **`test` and `valid` are pure functions of `trace_id`**, so growing the corpus only ever enlarges
  the train pool — the test set a run reports on is identical at every trace count, and the n=10 and
  n=1000 points are scored on exactly the same held-out steps.
- **The train subsample is nested**: for a fixed seed, the n=10 sample is a prefix of the n=20
  sample, so each step up the curve *adds* traces rather than redrawing — isolating corpus size from
  sample luck. Seeds vary the sample (and GEPA's search), giving the error bars.

This is a **3-way** split, unlike `wmh build` / `wmh bench` (2-way train/holdout): GEPA selecting and
being scored on the same set would make the curve optimistic, so the scaling number is reported on a
test set GEPA never touched.

### Two curves (`wmh/research/trace_scaling.py`)

`TraceScalingAblation` sweeps `mode × count`:

- **`base`** — the shipped `BASE_ENV_PROMPT`, scored on the test set with a retrieval buffer built
  from the N train traces. No GEPA: cheap, and isolates how far *retrieval alone* scales. Run this
  first to find the interesting range.
- **`gepa`** — GEPA optimizes the prompt on the N train traces (selecting on `valid`), then the
  winner is scored on the test set. The real "learning from more traces" curve — one GEPA run per
  count × seed, so it is the expensive one.

Counts are capped at the train pool, so the same sweep definition (`10,20,…,1000`) runs unchanged on
today's 66-trace tau2 corpus (collapsing to the few counts it can serve) and a future 1000-trace one.

## Running

```bash
# Cheap base curve first (no GEPA):
AWS_REGION=us-east-1 uv run python scripts/run_trace_scaling.py tau-bench \
    --counts 10,20,40 --modes base --seeds 0,1 --out scaling_base.json

# Then the GEPA learning curve:
AWS_REGION=us-east-1 uv run python scripts/run_trace_scaling.py tau-bench \
    --counts 10,20,40 --modes gepa --budget 12 --seeds 0,1 --out scaling_gepa.json
```

The runner takes a **benchmark name** (`tau-bench`, reusing its corpus + pinned judge from
`benchmarks/tau-bench/benchmark.toml`) or a raw `--file <trace.jsonl>`. It prints fidelity per
`(mode@count, seed)` and writes the full `AblationReport` JSON. Defaults: Bedrock Opus 4.8, offline
HashingEmbedder, `RubricJudge`.

**Extensible across benchmarks.** The ablation takes an ingested corpus + a base prompt, nothing tau2-
specific — `terminal-tasks` and `swe-bench` are just a different benchmark name (their corpora are
already committed under `examples/`). They are smaller today, so expect short curves until generated.

## Growing the corpus toward 1000

tau2 ships ~66 traces; terminal-tasks ~71; swe-bench ~21. To extend the curve, generate more traces
from the **real** benchmark and append to the corpus — the capture tooling is already in place:

- tau2: `tools/tau2-capture/` (run Sierra's real tau²-bench live on Bedrock, then `convert_to_wmh.py`
  → append to `examples/tau2-bench.otel.jsonl`). Sweep `tau2 run --num-tasks N` and concatenate. See
  that directory's README.
- terminal-tasks / swe-bench: the sibling `tools/terminal-tasks-capture/` and `tools/swe-bench-capture/`.

Because the split is hash-stable, appending traces never disturbs the existing test/valid bands — a
larger corpus simply deepens the train pool, so re-running the sweep extends the same curve.

## Reading the result

Each `(mode, count)` is a `ConditionReport` with `mean` ± `std` (across seeds). Plot `mean` vs.
`count` per mode; the `std` is the error bar. Read every gain against the **seed-stability band**
(experiment 1): a fidelity bump smaller than the across-seed std is noise. tau2 is a repetitive
airline/retail tool domain, so expect the curve to **saturate earlier** than a more diverse domain —
that saturation point is the headline.
