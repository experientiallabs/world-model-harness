# GEPA over RAG: knowledge accumulation on terminal-tasks

After studying the GEPA paper (arXiv 2507.19457) and `gepa-ai/gepa`, we corrected how we drive GEPA
and got a **large, real lift** — the kind RAG alone cannot produce.

## What we were doing wrong

GEPA's edge over plain retrieval is that it **distills reusable domain knowledge from training
traces into the prompt**. The library's own default reflector explicitly asks the model to capture
*"niche and domain-specific factual information"* and any *"generalizable strategy."* Our earlier
"surgical / preserve-verbatim / shortest-edit" reflection template **suppressed exactly that** — so
GEPA had nothing to add and returned a near-identical prompt (the "no lift" we kept seeing). We also
never enabled `use_merge` (combining complementary lessons from the Pareto frontier — a headline
GEPA feature).

## The fixes

- **Knowledge-accumulation reflection** (`_REFLECTION_PROMPT_TEMPLATE`): encourage the reflector to
  ADD concrete environment facts (output conventions, id/value patterns, error/empty behaviors) to a
  growing "Environment-specific notes" section, while preserving the existing hand-tuned rules
  (anti-regression kept; knowledge-suppression removed).
- **`use_merge=True`**: compose lessons two Pareto-front candidates each learned.
- **Multi-candidate sweep** (`scripts/gepa_multi.py`): pin base / base+RAG baselines, then spawn N
  GEPA candidates in parallel across different train subsets, each accumulating knowledge, ranked on
  the held-out test set. Fallback chain Opus 4.6 → 4.7 → Sonnet 4.6 → 4.8.
- Target **terminal-tasks / swe-bench** — RAG-solved tau2 has no headroom (base+RAG ≈ 0.96).

## Result (terminal-tasks, 1033→280-trace corpus, train/val/test)

**Validation fidelity (hard steps): base 0.718 → GEPA 0.838 (+0.12).** All four parallel candidates
independently climbed from ~0.72 into 0.835–0.839. This is the first large, reproducible GEPA lift
in the investigation — versus the ±0.005 we got on RAG-saturated tau2. (Test-set confirmation was
blocked by Bedrock throttling mid-run; validation is itself held out from reflection.)

## Why RAG can't do this: what GEPA actually learned

GEPA created an **"Environment-specific notes (accumulated knowledge)"** section with concrete rules
distilled from the traces — knowledge about *how the environment behaves*, not just similar
examples. Excerpts (`benchmarks/gepa-runs/terminal-evolved-prompt.txt`):

- **Shell dialect:** the env is `/bin/sh` (errors prefixed `sh: 1:`, not `bash:`).
- **Shell quoting:** `sort -t$'\t'` with a doubled backslash in the JSON args → `sort:
  multi-character tab` error + non-zero exit — don't assume the tab delimiter parsed.
- **Unauthenticated GitHub API:** returns HTTP 403 rate-limit JSON; downstream `jq .[0].name` →
  `Cannot index object with number`, and Python `data[:5]` → `TypeError: unhashable type: 'slice'`.
  Default to assuming rate-limiting; don't fabricate success.
- **Timeouts:** a `sleep 60` at a pipeline's start yields literally `(timed out)` with
  `is_error=true`, and **timeout takes priority** over any later pipeline error.
- **Pipelines:** the observation is the LAST command's stderr/stdout; a partial-success pipe still
  marks `is_error` when a component fails.

RAG can only retrieve a *similar past command*; it cannot teach the model "this environment is
`/bin/sh`, unauthenticated GitHub calls are rate-limited, timeouts print `(timed out)` and take
priority." That distilled, generalizable environment model is GEPA's contribution on top of
retrieval — exactly the tidbits/pointers we wanted it to accumulate.

## Reproduce

```bash
AWS_REGION=us-west-1 uv run python scripts/gepa_multi.py examples/terminal-tasks-full.otel.jsonl \
  --candidates 4 --iterations 6 --train 0.5 --val 0.25 --test-cap 40 \
  --out benchmarks/gepa-runs/terminal-multi.json
```

Note: Bedrock throttling makes full test scoring slow; the FallbackProvider degrades across the
model chain but a heavily-constrained window still crawls. Re-run when capacity is available to pin
the test-set number.
