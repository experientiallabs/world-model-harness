"""GEPA optimization-research harness.

A small, empirical surface for trying optimization directions and recording results — the seed of a
training-research surface as the harness grows from prompt optimization toward heavier methods. It
wraps the existing pipeline (`GEPAOptimizer`, `predict_observation`, the judge) rather than forking
it, so an experiment measures *what the harness actually does*.

Two layers:

- `ablation` — the framework: a `Condition` (a named set of knobs), an `Ablation` protocol
  (enumerate conditions + run one condition at one seed -> a scalar metric), and `run_ablation`,
  which sweeps every condition across multiple seeds and aggregates mean + std. Adding a new
  experiment = writing one `Ablation`.
- `pipeline` — the reusable build/eval primitives every ablation leans on: `optimize_prompt` (run
  GEPA at a chosen seed) and `score_prompt` (replay-score held-out fidelity via the canonical
  `wmh.engine.replay`, leak-free).

`seed_stability` is the first concrete experiment: how reproducible is GEPA's evolved prompt across
seeds. The train-vs-eval temperature sweep is parked (the shipped providers reject sampling params);
see docs/research_directions.md.
"""

from wmh.research.ablation import (
    Ablation,
    AblationReport,
    Condition,
    ConditionReport,
    SeedScore,
    aggregate,
    run_ablation,
)
from wmh.research.pipeline import optimize_prompt, score_prompt
from wmh.research.seed_stability import SeedStabilityAblation

__all__ = [
    "Ablation",
    "AblationReport",
    "Condition",
    "ConditionReport",
    "SeedScore",
    "SeedStabilityAblation",
    "aggregate",
    "optimize_prompt",
    "run_ablation",
    "score_prompt",
]
