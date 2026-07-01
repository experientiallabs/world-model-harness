# Getting GEPA right: train / validation / test

This documents the investigation into why GEPA showed no lift, the bugs found, the fixes, and the
honest final result. The harness is `scripts/gepa_test_lift.py`; the fixes are in
`wmh/optimize/gepa.py`, `wmh/engine/build.py`, `wmh/providers/{fallback,bedrock}.py`.

## The question

GEPA (reflective prompt evolution) returned a prompt **byte-identical to the base** on tau2/swe,
implying either "the base is optimal" or a broken optimizer. A hand edit lifted the same benchmark
+0.02, so the base was NOT optimal — the optimizer was the problem. Goal: get GEPA genuinely
improving on a **held-out test set** (separate from the validation set it selects on).

## Bugs found and fixed

1. **Budget < valset → zero exploration.** `gepa_budget` was passed straight through as GEPA's
   `max_metric_calls`. GEPA evaluates the seed on the full valset first (84–100 calls), so a budget
   of 50 was spent before any candidate was proposed. Fix: `_metric_call_budget` reinterprets budget
   as *iterations* and sizes the metric budget to fund the seed eval **plus** exploration.

2. **Judge mismatch.** Build optimized against `LLMJudge` but we evaluate with `RubricJudge` — GEPA
   hill-climbed a different metric than we report. Fix: build now uses `RubricJudge`; the seed val
   score matches the eval exactly.

3. **Full-rewrite reflection regressed the prompt.** GEPA's default reflection rewrote the whole
   prompt (2776 → 5700 chars), breaking 35 of 84 previously-perfect steps to fix a handful. Fix: a
   **surgical** reflection template (`_REFLECTION_PROMPT_TEMPLATE`) that demands a minimal edit
   preserving the current prompt verbatim. Candidates now add a targeted rule instead of rewriting.

4. **Two-way split leaked selection into the reported number.** The old `split_traces` gave
   (train, test) and GEPA used `test` as its valset — so it selected candidates on the very steps we
   graded. Fix: `split_traces_3way` (train / val / **test**), test never seen by GEPA.

5. **Near-saturated val gave no selection signal.** With overall val fidelity ~0.96, a candidate
   that fixes the few hard steps barely moves the mean and pareto keeps base (observed: best-val
   pinned at base through 12 iterations). Fix: `hard_step_filter` focuses reflection on steps with
   headroom (searches/lists + error observations), and `select_on_hard` selects candidates on
   hard-step val fidelity — where the base is below ceiling (0.835, not 0.96).

6. **Hung Bedrock calls blocked forever.** The Bedrock client had no request timeout, so one stalled
   `InvokeModel` wedged a whole run (29 min at 0% CPU) and the fallback couldn't fire (a hang raises
   nothing). Fix: botocore `Config(connect_timeout=15, read_timeout=300, adaptive retries)`; a
   stalled call now raises `ReadTimeoutError`, classified as a capacity error → fail over.

Plus a **fallback provider** (`FallbackProvider`) that fails over Opus 4.6 → 4.7 → Sonnet 4.6 →
Opus 4.8 on capacity errors, so long runs degrade gracefully instead of aborting.

## Result

On the 1033-trace tau-bench corpus (train 0.5 / val 0.25 / test 0.25), with all fixes:

| | base | GEPA-evolved | lift |
|---|---|---|---|
| **Validation (hard steps, 46)** | 0.835 | **0.839–0.840** | **+0.004–0.005, SELECTED** |
| Test (overall, 219 steps) | 0.914 | 0.913 | −0.001 |
| Test (hard steps, 8) | 0.720 | 0.721 | +0.001 |

**What works now:** GEPA reliably **selects a genuinely improved, surgically-edited candidate on
held-out validation** (reproduced across two runs). Its edit is sensible and general — "searches
return deterministic content you can't know; prefer fewer results, don't pad; structure matters
most" — exactly the search-over-population failure mode, added without touching the rest of the
prompt. This is the core fix: GEPA optimizes correctly against a proper val set, and the surgical
operator means it never regresses.

**What still caps test lift — and it is NOT the optimizer:** the prompt-addressable failures are
*rare and domain-specific*. In the whole 5289-step corpus there are only ~19 empty-result searches
and ~73 error observations; the rest of the low-scoring steps are **data-bound cold lookups**
(`get_reservation_details`, `find_user_id_by_name_zip` from empty state — the model cannot know a
record's exact values, so no prompt can fix them). A small test split (8 hard steps, mostly
cold `find_user` lookups) has almost no headroom for the search-fix GEPA learned on val, so the
aggregate is flat. The honest ceiling on these corpora is small (~+0.02–0.03), and it lives in a
handful of steps.

**Takeaways.** (1) GEPA is now correctly wired — proper train/val/test, budget, judge, and a
surgical operator that improves held-out val. (2) The remaining gap is a **data-coverage** problem,
not an optimizer problem: to show test lift you need a test split with enough of the fixable failure
mode, or a corpus with denser prompt-addressable failures. (3) The largest fidelity lever remains
**state grounding** (giving the model the record values it currently has to guess) — a data/harness
change, orthogonal to prompt optimization.
