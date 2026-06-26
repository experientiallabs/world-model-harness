"""Open-loop reconstruction-fidelity evaluation by replaying held-out steps ("open replay").

Replay is TEACHER-FORCED and so perfectly repeatable per step: for each held-out step we feed the
*real recorded* `(state_before, action)` and have the world model predict the observation, then
score it against the *real recorded* observation. Nothing the model generates feeds forward, so a
bad prediction at one step never contaminates another — the score isolates per-step fidelity. (The
closed-loop counterpart, where predictions feed forward, is a future direction: see
docs/closed_loop.md.)

Because the world model can run at temperature > 0, each step is sampled over `rollouts` predictions
and scored independently, yielding a per-step distribution (mean + std). The judge is pluggable;
`RubricJudge` (the Qwen-AgentWorld-style 5-dimension scorer) is the default for evaluation.

Retrieval mirrors serving and GEPA: leak-free demos from the TRAIN corpus only, never the query
step's own trace.
"""

from __future__ import annotations

import random
from statistics import fmean, pstdev

from pydantic import BaseModel, Field

from wmh.core.render import render_action
from wmh.core.types import Step, Trace
from wmh.optimize.gepa import predict_observation
from wmh.optimize.judge import Judge
from wmh.providers.base import Provider
from wmh.retrieval import Retriever
from wmh.retrieval.leakfree import DemoRetriever

# Turns scored per trace when sample_turns="sampled", following Qwen-AgentWorld's protocol:
# first, last, and 3 uniformly-sampled intermediate turns.
SAMPLED_TURNS = 5


class StepResult(BaseModel):
    """One replayed step scored over one or more rollouts.

    `score` is the headline (mean across rollouts) so existing readers keep working; `scores` holds
    the per-rollout values and `score_std` their spread. `dimensions` is the per-dimension mean when
    a rubric judge is used (empty otherwise).
    """

    trace_id: str
    task: str | None = None
    action: str  # rendered action, for human-readable scorecards
    actual: str
    predicted: str  # the first rollout's prediction (representative, for the scorecard)
    score: float  # mean judge score across rollouts, 0..1
    score_std: float = 0.0
    scores: list[float] = Field(default_factory=list)  # per-rollout judge scores
    dimensions: dict[str, float] = Field(default_factory=dict)  # per-dimension mean (rubric judge)
    critique: str = ""  # first rollout's critique
    is_error_actual: bool = False
    is_error_predicted: bool = False  # first rollout's is_error


class ReplayReport(BaseModel):
    """Aggregate fidelity over a replay run."""

    mean_score: float = 0.0
    score_std: float = 0.0  # std of per-step mean scores across steps
    error_flag_accuracy: float = 0.0  # fraction where predicted is_error matched actual
    n_steps: int = 0
    rollouts: int = 1
    results: list[StepResult] = Field(default_factory=list)

    def summary(self) -> str:
        return (
            f"fidelity={self.mean_score:.3f}±{self.score_std:.3f} "
            f"error_flag_acc={self.error_flag_accuracy:.3f} "
            f"n={self.n_steps} rollouts={self.rollouts}"
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
    rollouts: int = 1,
    temperature: float = 0.0,
    sample_turns: str = "all",
    seed: int = 0,
) -> ReplayReport:
    """Replay held-out steps, scoring predicted vs. actual observations.

    - `rollouts` × `temperature`: sample each step `rollouts` times at `temperature` (use >0 for a
      real distribution) and score each; the step's score is the mean, with std reported.
    - `sample_turns`: "all" scores every step; "sampled" scores first/last/3-uniform per trace
      (Qwen-AgentWorld's 5-turn protocol) using `seed` for reproducible turn selection.
    - `retriever` + `train` enable leak-free RAG (demos from the train corpus, never the own trace);
      omit either for zero-shot.
    """
    demos = DemoRetriever(retriever, train or [], top_k=top_k)
    rng = random.Random(seed)
    results: list[StepResult] = []
    for trace in held_out:
        for step in _select_steps(trace, sample_turns, rng):
            results.append(
                _score_step(
                    prompt, trace.trace_id, step, provider, judge, demos, rollouts, temperature
                )
            )
    return _aggregate(results, rollouts)


def _select_steps(trace: Trace, sample_turns: str, rng: random.Random) -> list[Step]:
    """Pick which steps of `trace` to score. 'all' = every step; 'sampled' = first/last/3 middle."""
    steps = trace.steps
    if sample_turns != "sampled" or len(steps) <= SAMPLED_TURNS:
        return steps
    middle = list(range(1, len(steps) - 1))
    picks = sorted(rng.sample(middle, SAMPLED_TURNS - 2))
    indices = [0, *picks, len(steps) - 1]
    return [steps[i] for i in indices]


def _score_step(
    prompt: str,
    trace_id: str,
    step: Step,
    provider: Provider,
    judge: Judge,
    demos: DemoRetriever,
    rollouts: int,
    temperature: float,
) -> StepResult:
    """Sample `rollouts` predictions for a step and score each against the recorded observation."""
    step_demos = demos.demos_for(trace_id, step)
    scores: list[float] = []
    dim_sums: dict[str, float] = {}
    first_predicted = None
    first_critique = ""
    for _ in range(max(1, rollouts)):
        predicted = predict_observation(
            provider,
            prompt,
            step.task,
            step.state_before,
            step.action,
            demos=step_demos,
            temperature=temperature,
        )
        verdict = judge.score(predicted, step.observation, step)
        scores.append(verdict.score)
        for dim, val in verdict.dimensions.items():
            dim_sums[dim] = dim_sums.get(dim, 0.0) + val
        if first_predicted is None:
            first_predicted, first_critique = predicted, verdict.critique
    assert first_predicted is not None  # rollouts >= 1
    n = len(scores)
    return StepResult(
        trace_id=trace_id,
        task=step.task,
        action=render_action(step.action),
        actual=step.observation.content,
        predicted=first_predicted.content,
        score=fmean(scores),
        score_std=pstdev(scores) if n > 1 else 0.0,
        scores=scores,
        dimensions={dim: total / n for dim, total in dim_sums.items()},
        critique=first_critique,
        is_error_actual=step.observation.is_error,
        is_error_predicted=first_predicted.is_error,
    )


def _aggregate(results: list[StepResult], rollouts: int) -> ReplayReport:
    if not results:
        return ReplayReport(rollouts=rollouts)
    step_means = [r.score for r in results]
    error_acc = fmean(1.0 if r.is_error_predicted == r.is_error_actual else 0.0 for r in results)
    return ReplayReport(
        mean_score=fmean(step_means),
        score_std=pstdev(step_means) if len(step_means) > 1 else 0.0,
        error_flag_accuracy=error_acc,
        n_steps=len(results),
        rollouts=rollouts,
        results=results,
    )
