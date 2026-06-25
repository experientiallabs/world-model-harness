"""GEPA prompt optimization + the LLM judge that scores predictions.

GEPA (arXiv 2507.19457): replay held-out steps through a candidate prompt, score predicted vs. real
observation with an LLM judge that also returns a natural-language critique, reflect on the critiques
to mutate the prompt, and keep a Pareto frontier of candidates across trace buckets.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from wmh.providers.base import Provider
from wmh.types import Observation, Step, Trace


class JudgeResult(BaseModel):
    score: float  # 0..1 semantic match of predicted vs. actual observation
    critique: str  # natural-language feedback; feeds GEPA reflection


@runtime_checkable
class Judge(Protocol):
    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        ...


class LLMJudge:
    """Opus-based semantic-match judge (default fitness signal)."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        # TODO: prompt the judge to compare functional equivalence; parse score + critique.
        raise NotImplementedError


class OptimizeResult(BaseModel):
    prompt: str  # winning specialized env prompt
    frontier: list[str] = Field(default_factory=list)  # Pareto candidates
    metrics: dict[str, float] = Field(default_factory=dict)  # held-out accuracy, judge agreement


@runtime_checkable
class Optimizer(Protocol):
    def optimize(
        self, train: list[Trace], test: list[Trace], base_prompt: str, budget: int
    ) -> OptimizeResult:
        ...


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
