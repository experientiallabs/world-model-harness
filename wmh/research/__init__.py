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
  GEPA at a chosen rollout temperature + seed) and `score_prompt` (replay-score held-out fidelity at
  a chosen eval temperature, leak-free).

`temperature` is the first concrete experiment: train-vs-eval temperature (docs/gepa_research.md).
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
from wmh.research.temperature import (
    TEMPERATURE_CONDITIONS,
    TemperatureAblation,
    temperature_conditions,
)

__all__ = [
    "TEMPERATURE_CONDITIONS",
    "Ablation",
    "AblationReport",
    "Condition",
    "ConditionReport",
    "SeedScore",
    "TemperatureAblation",
    "aggregate",
    "optimize_prompt",
    "run_ablation",
    "score_prompt",
    "temperature_conditions",
]
