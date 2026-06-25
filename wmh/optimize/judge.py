"""The LLM judge that scores a predicted observation against the real one.

The judge is GEPA's fitness signal: it returns a scalar score *and* a natural-language critique,
and the critique is what GEPA reflects on to mutate the prompt.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from wmh.core.types import Observation, Step
from wmh.providers.base import Provider


class JudgeResult(BaseModel):
    score: float  # 0..1 semantic match of predicted vs. actual observation
    critique: str  # natural-language feedback; feeds GEPA reflection


@runtime_checkable
class Judge(Protocol):
    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult: ...


class LLMJudge:
    """Opus-based semantic-match judge (default fitness signal)."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        # TODO: prompt the judge to compare functional equivalence; parse score + critique.
        raise NotImplementedError
