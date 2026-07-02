# financebench

Financial-document QA over real SEC-filing evidence excerpts. The environment is a workspace
whose `docs/` holds the task's true evidence doc(s) plus 4 distractors; the agent retrieves with
real shell commands and submits an answer. Scoring is deterministic (numeric match, token-F1
fallback) — see `environment_capture/benchmarks/financebench.py`.

## Contents

- `data/train.jsonl` (121 tasks) / `data/test.jsonl` (5 tasks) — agent-visible tasks
  (prompt + doc ids + difficulty stratum).
- `corpus/<doc_id>.txt` — 164 evidence excerpts (verbatim upstream `evidence_text`).
- `gold/<task_id>.json` — gold answers (`answer` text + parsed `numeric`), never staged into the
  agent workspace.
- `traces.otel.jsonl` — the trace corpus: **89 traces / 139 real transitions** (train split
  only; the hidden test split is never captured so the world model can't absorb its dynamics).
- `convert_cache.py` — the converter that produced the corpus (see provenance).
- `capture.py` — fresh real-run capture against this adapter (Bedrock agent), used to top up the
  corpus with richer multi-step trajectories.
- `evals/default.toml` — fidelity suite; run with
  `uv run wmh eval run financebench/default --examples-root environment-capture`.
- `wm_replace_demo.py` — the same agent runs the held-out test tasks against the REAL env and a
  world model of it (`wmh build --name financebench --file .../traces.otel.jsonl` first), graded
  by the same deterministic grader; full transcripts land in `runs/` for auditing.

## Results (2026-07-02, corpus as committed)

- **Open-loop fidelity** (suite `financebench/default`, seed 0, Opus 4.8 target + rubric judge):
  mean fidelity **0.581**, error-flag accuracy **0.800**, n=35 held-out steps. Notably below the
  shell-like corpora (tau ~0.90, terminal ~0.86, swe ~0.82): observations here are long verbatim
  document excerpts, which are much harder to reconstruct than command output.
- **WM-replacement demo** (5 test tasks, Opus 4.8 agent, Opus 4.8 WM): reward agreement 5/5 on
  the first run; a 2-task audit rerun showed agent-side nondeterminism in the REAL env (a task
  flipping 1.0→0.0 across runs), so single-run agreement on n=5 is indicative, not a claim.
  Audited caveat: the WM invents plausible-but-wrong doc filenames yet often reconstructs
  historically TRUE figures — the backbone knows public SEC facts from pretraining, so
  financebench agreement partly reflects model knowledge rather than trace grounding. Treat this
  benchmark as an easy-mode WM target; the stateful benchmarks are the stronger test.

## Provenance

- **Dataset**: [PatronusAI/financebench](https://huggingface.co/datasets/PatronusAI/financebench),
  filtered to rows gradeable deterministically offline; evidence text is from public SEC filings
  (10-K/10-Q/earnings). Task ids, doc staging (evidence + 4 distractors), and train/test split come
  from a prior materialization of the upstream dataset, reused as data; all adapter/grader code
  here is fresh.
- **Traces**: converted with `convert_cache.py` from a frozen baseline cache of REAL runs over
  the same materialization (model `gpt-5.4`, mean reward 0.289 across the full 121-task train
  split; 32 zero-transition trajectories skipped at conversion). Converted traces keep the
  original run's reward in metadata; future fresh captures via `capture.py` are graded by this
  adapter's grader (documented thresholds, not identical) and carry their own model id.

## License — read before redistributing

FinanceBench is published under **CC BY-NC 4.0** (non-commercial, attribution). The task data,
evidence corpus, and any traces embedding evidence text are redistributed here for
**non-commercial benchmark/research use, with attribution to PatronusAI**. Do not use this data
commercially.
