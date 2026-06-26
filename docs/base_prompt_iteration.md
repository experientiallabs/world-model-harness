# Iterating on BASE_ENV_PROMPT with replay fidelity

`BASE_ENV_PROMPT` (`wmh/engine/prompts.py`) is layer (a): the env-agnostic prompt GEPA evolves from.
We want it both **general** (works across domains) and a **strong GEPA starting surface** (high
zero-/few-shot reconstruction fidelity before any evolution). We tune it by measuring, not guessing.

## The measurement loop

`wmh eval` (engine: `wmh/engine/eval.py` → `wmh/engine/replay.py`) replays each trace file's held-out
split through a prompt and scores predicted vs. real observations with the `LLMJudge` (0..1
functional equivalence) plus a deterministic is_error-flag check. Run:

```bash
AWS_REGION=us-east-1 uv run wmh eval \
  examples/tau2-bench.otel.jsonl \
  --out report.json
```

`wmh eval` takes one or more trace files; pass every benchmark corpus you have captured. Today the
committed corpus is `examples/tau2-bench.otel.jsonl` only (see [`benchmarks.md`](./benchmarks.md));
add more files here as new benchmarks are captured via the `tools/<bench>-capture/` pattern.

Inspect the lowest-scoring steps' critiques to find systematic failure modes, change the prompt,
re-run on the same split, and keep changes that move fidelity without overfitting to one benchmark.

## What the first baseline taught us

Initial overall fidelity was ~0.43, but inspecting results surfaced two distinct issues:

1. **A data artifact, not a model failure.** Half the held-out steps had *empty* ground truth —
   they were the agent's final submit/answer turn, which has no environment reply. Scoring against a
   non-existent observation dragged the number down. Fix: trace ingestion only emits a step for an
   agent turn that HAS a following environment reply (unpaired trailing turns are dropped). On steps
   with a real observation, fidelity was already ~0.62.

2. **Real, fixable model failure modes** (drove the prompt rewrite):
   - *Fabricating concrete data* the environment alone knows (e.g. inventing SQL result rows instead
     of the DB's actual contents).
   - *Inventing stdout* when the real command prints nothing (assignments, writes, redirects).
   - *Guessing success vs. error wrong* (heredoc syntax errors predicted as success, and vice-versa).

## The resulting base prompt

The rewritten `BASE_ENV_PROMPT` targets those three failure modes while staying domain-agnostic, and
adds explicit **stay-in-character** guidance for edge cases: a terminal env must answer a stray "hi"
with `command not found` (not a chat reply); an API env must answer an unknown call with that API's
own error shape. It instructs the model to ground values in STATE/HISTORY (never fabricate), to
output only the bytes that actually reach the agent (empty when nothing prints), and to decide
success/error from what the action would really do.

## Notes

- This is deliberately NOT GEPA-on-the-base: we keep the base general and hand-tuned, then let GEPA
  specialize per-project from this stronger starting point.
- Fidelity numbers on a small corpus are noisy; weight the overall step-weighted mean, and treat
  per-file numbers as a signal rather than a benchmark until more benchmarks are captured. (The
  baseline above was measured on the earlier, larger multi-benchmark corpus; the methodology is
  unchanged — only the committed corpus is now tau2-only.)
```
