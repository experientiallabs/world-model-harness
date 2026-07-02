# dabstep

Data-analysis QA over a shared payments dataset and a business-rules manual. Each task is a
question whose correct answer requires reading `manual.md` (it defines what "authorized", "fee",
and "fraud rate" mean — the raw columns are ambiguous on their own) and computing over the CSV/JSON
context files with real shell + pandas. The environment stages the task's context files into a
fresh workspace's `./data/` directory; the agent explores, analyzes, and submits an answer. Scoring
is deterministic (numeric tolerance 0.01, normalized string/list match, accepted alternates) — see
`environment_capture/benchmarks/dabstep.py`.

## Contents

- `data/train.jsonl` (5 tasks) / `data/test.jsonl` (5 tasks) — agent-visible tasks
  (prompt + `file_ids` + difficulty level). Train and test ids are disjoint.
- `datafiles/<file_id>` — the shared context files, committed **except** `payments.csv`
  (~23 MB, gitignored): `manual.md`, `fees.json`, `acquirer_countries.csv`,
  `merchant_category_codes.csv`, `merchant_data.json`, `payments-readme.md`.
- `gold/<task_id>.json` — gold answers (`answer` + optional `numeric` + `accept` variants), never
  staged into the agent workspace.
- `traces.otel.jsonl` — the trace corpus: **28 traces / 293 real transitions** (train split only;
  the hidden test split is never captured so the world model can't absorb its dynamics).
- `fetch_data.py` — downloads the gitignored `payments.csv` (and, with `--all`, every context file)
  from the upstream HuggingFace dataset, so a fresh clone is runnable.
- `convert_cache.py` — the converter that seeded the corpus from a frozen baseline cache of real
  runs (see provenance).
- `capture.py` — fresh real-run capture against this adapter (Bedrock agent), used to grow the
  corpus with richer multi-step trajectories.

## Running it

```bash
# 1. pull the large context file (payments.csv is gitignored)
uv run python environment-capture/dabstep/fetch_data.py

# 2. capture fresh real runs on Bedrock (each model runs the full train split)
uv run python environment-capture/dabstep/capture.py \
    --models us.anthropic.claude-opus-4-8,us.anthropic.claude-opus-4-7 --runs 1 \
    --out environment-capture/dabstep/traces.otel.jsonl --append
```

## Provenance

- **Dataset**: [adyen/DABstep](https://huggingface.co/datasets/adyen/DABstep). The committed
  `data/*.jsonl` + `gold/*.json` are the real DABstep `dev` set (the 10 gradeable questions —
  the `default` split is server-scored with no local gold), split disjointly into 5 train + 5 test.
  Task ids, `file_ids`, the train/test split, and the context files come from a prior
  materialization of the upstream dataset, reused as data; all adapter/grader code here is fresh
  (the documented thresholds are not inherited from anywhere).
- **Traces**: seeded with `convert_cache.py` from a frozen baseline cache of REAL runs over the
  same materialization (**3 traces**, model `gpt-5.4`, mean reward 0.0 — the bare baseline agent's
  heredoc quoting failed before it could submit; 2 zero-transition trajectories skipped at
  conversion), then grown with `capture.py` — **25 fresh real runs** on Bedrock
  (`us.anthropic.claude-opus-4-8` and `-4-7`, two passes each; `-4-6-v1`, one pass — mean reward
  0.240, solving `dab-train-0` and `dab-train-3`). Every train task is covered by every model.
  Converted traces keep the original run's reward; fresh captures are graded by this adapter's
  grader and carry their own model id. Multi-run trace ids never collide: a fresh run's task id is
  suffixed with its model + run tag (e.g. `dab-train-3#opus48-r1`).
- The recording harness echoed an ALLCAPS `*_SUBMIT` sentinel into the baseline runs' final
  command/output to mark the submission; it is normalized to the neutral `SUBMIT` at conversion
  (apparatus protocol, not environment content — no result, path, or number is altered). Fresh
  Bedrock captures use a real `submit` tool and carry no such sentinel.

## License — read before redistributing

DABstep is published under **CC BY 4.0** (attribution). The task data, context files, gold answers,
and traces are redistributed here under CC BY 4.0 **with attribution to Adyen**
(https://huggingface.co/datasets/adyen/DABstep). Redistribution and commercial use are permitted
provided attribution is retained.
