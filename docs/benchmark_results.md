# Benchmark results: reproducibility

The headline numbers in the README's *Benchmark results* section come from open-loop reconstruction
fidelity (`wmh eval`) on the committed `examples/*.otel.jsonl` corpora (tau2-bench, terminal-tasks,
swe-bench, captured from the upstream benchmarks). This doc records the exact methodology so the
numbers can be regenerated.

## The 5-baseline × 3-corpus grid

The README table sweeps five baselines across three corpora. Reproduce the whole grid with the
committed runners (`--train-split 0.7 --seed 0`, all held-out turns, rubric judge = Bedrock Opus
4.8 for every row):

```bash
# Baselines 1-4: Opus 4.8 prompted as the environment. Per corpus, four cells —
#   base / base+RAG / GEPA / GEPA+RAG  (--no-rag toggles retrieval; --prompt swaps the GEPA prompt).
AWS_REGION=us-west-1 uv run wmh eval examples/tau2-bench.otel.jsonl \
  --region us-west-1 --judge rubric --train-split 0.7 --seed 0 --no-rag \
  --out benchmarks/results/grid-tau2-bench-base-norag.json
# ...drop --no-rag for base+RAG; add --prompt benchmarks/gepa-v2-prompts/tau2-bench.optimized.txt
# for the GEPA cells. Repeat for terminal-tasks and swe-bench.

# Baseline 5: Qwen-AgentWorld-35B-A3B as the world model (vLLM, OpenAI-compatible), Opus judge.
# The world model and judge are split apart (wmh eval couples them) by scripts/eval_agentworld.py.
OPENAI_BASE_URL=http://localhost:8001/v1 OPENAI_API_KEY=dummy AWS_REGION=us-west-1 \
  uv run python scripts/eval_agentworld.py examples/tau2-bench.otel.jsonl \
  --train-split 0.7 --seed 0 --max-tokens 4096 \
  --out benchmarks/results/grid-tau2-bench-agentworld-rag.json
```

The GEPA prompts were rebuilt on the *current* base prompt (`wmh build --train-split 0.7
--gepa-budget 50`); for tau2-bench and swe-bench GEPA returned a prompt byte-identical to the base
(it found no improvement), saved under `benchmarks/gepa-v2-prompts/`. Render the plot + markdown
table from the committed per-step reports:

```bash
uv run --extra viz python scripts/plot_baseline_grid.py \
  --results-dir benchmarks/results --out docs/img/baseline_grid.png \
  --table-out docs/baseline_grid_table.md
```

**AgentWorld notes.** It is a reasoning model: it emits a hidden think-trace before the JSON
observation, so the eval default (`max_tokens=1024`) truncates it to an empty string — use ≥4096
(measured generations are ~600-1000 completion tokens, `finish=stop`). Served via vLLM with
`--max-num-seqs` ≥ the number of concurrent corpus streams so they batch. The judge stays Opus 4.8
so the AgentWorld fidelity is directly comparable to the Opus rows.

## tau2-only deep dive (historical: base vs GEPA on the old base prompt)

The numbers below predate the base-prompt hand-tuning and the 5-baseline grid; they document the
*original* GEPA lift on the old base prompt, kept for provenance.

## Reproduce

Requires Bedrock credentials (Opus 4.8 is the only live backend here). The runs cost roughly
$1–2 each and take a few minutes (84 held-out steps × judge calls).

```bash
# Base prompt (the un-evolved BASE_ENV_PROMPT)
AWS_REGION=us-east-1 uv run wmh eval examples/tau2-bench.otel.jsonl \
  --region us-east-1 --judge rubric --train-split 0.7 --seed 0 \
  --out base_report.json

# GEPA-optimized prompt (the committed canonical model)
AWS_REGION=us-east-1 uv run wmh eval examples/tau2-bench.otel.jsonl \
  --region us-east-1 --prompt world-models/tau-telecom/prompts/optimized.txt \
  --judge rubric --train-split 0.7 --seed 0 \
  --out optimized_report.json
```

`--train-split 0.7 --seed 0` deterministically selects the same 11-trace / 84-step held-out split
both times, so the two runs are comparable. Each `*_report.json` carries per-step scores, per-step
rubric dimensions, and the judge critiques.

## Results obtained (2026-06, Bedrock Opus 4.8, top-k=5 retrieval)

The committed per-step reports are in `benchmarks/results/tau2-{base,optimized}.json` (each step's
predicted vs. actual observation, the 5 rubric dimensions, and the judge critique).

| Prompt | held-out steps | fidelity (mean ± std) | error-flag acc |
|---|---|---|---|
| Base | 84 | ~0.74 ± 0.35 | ~0.80 |
| GEPA-optimized | 84 | ~0.86 ± 0.20 | ~1.00 |

Per-dimension (rubric judge), optimized prompt: format ~0.99, factuality ~0.72, consistency ~0.88,
realism ~0.97, quality ~0.76.

**On variance / repeatability (multi-run hardening).** The LLM judge is non-deterministic, so the
same split scores slightly differently run to run. Repeating both evals on the identical 84-step
holdout:

| Prompt | run 1 | run 2 | mean ± std |
|---|---|---|---|
| Base | 0.755 | 0.723 | 0.739 ± 0.016 |
| GEPA-optimized | 0.864 | 0.854 | 0.859 ± 0.005 |

The two distributions **do not overlap** (worst optimized 0.854 > best base 0.755), so the
**+0.12 lift is stable, not run-to-run luck**. Treat the headline table as approximate (≈±0.02
cross-run on top of the per-step std). The committed report JSONs are run 2 (base 0.723, optimized
0.854). Both runs use the same single seed (`--seed 0`), so this measures judge non-determinism on
one split — not seed-to-seed variance, which remains a GEPA-research follow-up.

## Caveats

- **One corpus, 84-step holdout** (per-step std ±0.19–0.34; cross-run ≈±0.02). Directional, not a
  leaderboard. More benchmarks/larger holdouts would tighten it further.
- The judge is an LLM (Opus 4.8) at temperature 0, but still has some variance; the per-step scores
  in the committed reports are a single sample each.
- Retrieval uses the offline lexical `HashingEmbedder` (semantic phi untested).
- `held_out_accuracy` in `world-models/tau-telecom/metrics.json` (0.675) is GEPA's *internal*
  validation score over its own 317-rollout search — a different measurement from these `wmh eval`
  fidelity numbers; don't conflate them.
