"""Reconstruction-fidelity evaluation by replaying held-out steps ("open replay").

Given held-out traces, replay each step's `(state, action)` through a world-model prompt and score
the predicted observation against the *real* recorded observation with the `LLMJudge`. Produces a
per-step record and per-benchmark aggregates — a scorecard of how faithfully the world model
reconstructs the real environment.

This is the measurement loop used to evaluate and iterate on `BASE_ENV_PROMPT`. It mirrors serving:
predictions use the shared `predict_observation` (same prompt assembly) and the same DreamGym
retrieval, with the same leak-free rule GEPA uses — retrieve from the TRAIN corpus only and never
from the query step's own trace.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from wmh.core.render import render_action
from wmh.core.types import Trace
from wmh.optimize.gepa import predict_observation
from wmh.optimize.judge import Judge
from wmh.providers.base import Provider
from wmh.retrieval import Retriever
from wmh.retrieval.leakfree import DemoRetriever


class StepResult(BaseModel):
    """One replayed step: model prediction vs. recorded truth, plus the judge's verdict."""

    trace_id: str
    task: str | None = None
    action: str  # rendered action, for human-readable scorecards
    predicted: str
    actual: str
    score: float  # judge functional-equivalence score, 0..1
    critique: str
    is_error_actual: bool
    is_error_predicted: bool


class ReplayReport(BaseModel):
    """Aggregate fidelity over a replay run."""

    mean_score: float = 0.0
    error_flag_accuracy: float = 0.0  # fraction where predicted is_error matched actual
    n_steps: int = 0
    results: list[StepResult] = Field(default_factory=list)

    def summary(self) -> str:
        return (
            f"fidelity={self.mean_score:.3f} error_flag_acc={self.error_flag_accuracy:.3f} "
            f"n={self.n_steps}"
        )


def replay(
    prompt: str,
    held_out: list[Trace],
    provider: Provider,
    judge: Judge,
    *,
    retriever: Retriever | None = None,
    train: list[Trace] | None = None,
    top_k: int = 5,
) -> ReplayReport:
    """Replay every step of `held_out`, scoring predicted vs. actual observations.

    `retriever` + `train` enable RAG (retrieve demos from the train corpus, leak-free). When either
    is None, replay is zero-shot. Returns a `ReplayReport` with per-step results and aggregates.
    """
    demos = DemoRetriever(retriever, train or [], top_k=top_k)
    results: list[StepResult] = []
    for trace in held_out:
        for step in trace.steps:
            predicted = predict_observation(
                provider,
                prompt,
                step.task,
                step.state_before,
                step.action,
                demos=demos.demos_for(trace.trace_id, step),
            )
            verdict = judge.score(predicted, step.observation, step)
            results.append(
                StepResult(
                    trace_id=trace.trace_id,
                    task=step.task,
                    action=render_action(step.action),
                    predicted=predicted.content,
                    actual=step.observation.content,
                    score=verdict.score,
                    critique=verdict.critique,
                    is_error_actual=step.observation.is_error,
                    is_error_predicted=predicted.is_error,
                )
            )
    return _aggregate(results)


def _aggregate(results: list[StepResult]) -> ReplayReport:
    if not results:
        return ReplayReport()
    n = len(results)
    mean_score = sum(r.score for r in results) / n
    error_acc = sum(1 for r in results if r.is_error_predicted == r.is_error_actual) / n
    return ReplayReport(
        mean_score=mean_score, error_flag_accuracy=error_acc, n_steps=n, results=results
    )
