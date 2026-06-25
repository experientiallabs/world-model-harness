"""GEPA reflective prompt evolution.

GEPA (arXiv 2507.19457): replay held-out steps through a candidate prompt, score predicted vs.
real observation with the LLM judge (which also returns a natural-language critique), reflect on
those critiques to mutate the prompt, and keep a Pareto frontier of candidates across trace buckets.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from wmh.core.types import Trace
from wmh.optimize.judge import Judge
from wmh.providers.base import Provider


class OptimizeMetrics(BaseModel):
    """Outcome metrics from an optimization run."""

    held_out_accuracy: float = 0.0  # mean judge score on the held-out split
    judge_agreement: float = 0.0  # judge self-consistency / human-agreement proxy
    rollouts_used: int = 0


class OptimizeResult(BaseModel):
    prompt: str  # winning specialized env prompt
    frontier: list[str] = Field(default_factory=list)  # Pareto candidates
    metrics: OptimizeMetrics = Field(default_factory=OptimizeMetrics)


@runtime_checkable
class Optimizer(Protocol):
    def optimize(
        self, train: list[Trace], test: list[Trace], base_prompt: str, budget: int
    ) -> OptimizeResult: ...


class GEPAOptimizer:
    """Reflective prompt evolution against the held-out trace split."""

    def __init__(self, provider: Provider, judge: Judge) -> None:
        self._provider = provider
        self._judge = judge

    def optimize(
        self, train: list[Trace], test: list[Trace], base_prompt: str, budget: int
    ) -> OptimizeResult:
        # TODO: candidate pool -> replay -> judge -> reflect/mutate -> Pareto update, within budget.
        raise NotImplementedError
