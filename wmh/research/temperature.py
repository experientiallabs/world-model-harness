"""Train-vs-eval temperature ablation — the harness's first concrete experiment.

`predict_observation` historically hardcoded `temperature=0.0`. This experiment crosses the rollout
temperature applied during TRAINING (GEPA's candidate evaluation + reflection rollouts) with the one
applied during EVALUATION (held-out replay scoring), to ask: does optimizing/serving
deterministically (T=0) beat varying it (T=1) — and is the effect a *training* effect, an
*evaluation* effect, or both?

The 2×2 grid (T_train ∈ {0,1}) × (T_eval ∈ {0,1}) is run across multiple seeds; `run_ablation`
reports each cell's mean ± std. See docs/gepa_research.md for how to read the result.

This module is dependency-injected: it takes a factory for the (provider, judge, embedder) trio and
the already-split train/held-out traces, so the unit test drives it with fakes and the `scripts/`
runner drives it with live Bedrock — same code path.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from wmh.core.types import JsonValue, Trace
from wmh.optimize.judge import Judge
from wmh.providers.base import Embedder, Provider
from wmh.research.ablation import Condition
from wmh.research.pipeline import optimize_prompt, score_prompt

# The provider/judge/embedder a single run uses. A factory (not a shared instance) so per-run cost
# tracking or stateful fakes can be isolated; `embedder=None` means zero-shot (no RAG).
Backends = tuple[Provider, Judge, Embedder | None]
BackendFactory = Callable[[], Backends]

# Param keys this ablation owns inside a Condition.
TRAIN_TEMP = "train_temperature"
EVAL_TEMP = "eval_temperature"


def temperature_conditions(temps: Sequence[float] = (0.0, 1.0)) -> list[Condition]:
    """The crossed grid of (train, eval) temperatures, labelled `Ttrain=../Teval=..`."""
    return [
        Condition(
            label=f"Ttrain={t_train:g}/Teval={t_eval:g}",
            params={TRAIN_TEMP: t_train, EVAL_TEMP: t_eval},
        )
        for t_train in temps
        for t_eval in temps
    ]


# The default experiment grid: deterministic vs. variable on each axis.
TEMPERATURE_CONDITIONS = temperature_conditions()


class TemperatureAblation:
    """Cross train-rollout temperature with eval-rollout temperature; metric = held-out fidelity."""

    name = "train-vs-eval-temperature"

    def __init__(
        self,
        train: list[Trace],
        held_out: list[Trace],
        base_prompt: str,
        *,
        make_backends: BackendFactory,
        budget: int,
        conditions: Sequence[Condition] | None = None,
        top_k: int = 5,
    ) -> None:
        self._train = train
        self._held_out = held_out
        self._base_prompt = base_prompt
        self._make_backends = make_backends
        self._budget = budget
        self._conditions = list(conditions) if conditions is not None else TEMPERATURE_CONDITIONS
        self._top_k = top_k

    def conditions(self) -> list[Condition]:
        return self._conditions

    def run(self, condition: Condition, seed: int) -> float:
        """Build at this condition's train temperature + `seed`, then score at its eval temperature.

        The metric is held-out reconstruction fidelity (mean judge score, 0..1) — the same number
        `wmh eval` reports — so cells are directly comparable to the rest of the harness.
        """
        train_temp = _as_float(condition.params[TRAIN_TEMP])
        eval_temp = _as_float(condition.params[EVAL_TEMP])
        provider, judge, embedder = self._make_backends()

        result = optimize_prompt(
            self._train,
            self._held_out,
            self._base_prompt,
            provider=provider,
            judge=judge,
            embedder=embedder,
            budget=self._budget,
            train_temperature=train_temp,
            seed=seed,
        )
        return score_prompt(
            result.prompt,
            self._held_out,
            provider=provider,
            judge=judge,
            embedder=embedder,
            train=self._train,
            eval_temperature=eval_temp,
            top_k=self._top_k,
        )


def _as_float(value: JsonValue) -> float:
    """Coerce a Condition param to float, rejecting bools and non-numerics."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"temperature param must be a number, got {value!r}")
    return float(value)
