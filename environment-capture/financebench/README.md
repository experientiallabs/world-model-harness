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
