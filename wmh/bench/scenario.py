"""Timed OPEN-LOOP scenario replay through a world model (the eval predict path, with timing).

`wmh bench scenario <name>` is the world-model half of a scenario comparison. It replays a recorded
trace's `(state, action)` steps through the model **teacher-forced — exactly as `wmh eval` /
`wmh.engine.replay` do**: each step predicts from the RECORDED `state_before` (not the model's own
prior predictions) with leak-free demos retrieved from the train corpus. It is open-loop on purpose
— the same measurement the evaluator runs — but it adds per-step wall-clock timing and LLM
tokens/cost, and prints each predicted observation as it lands. The real environment runs the SAME
scenario closed-loop (see the `tools/<benchmark>-capture/` runners); you compare the two end times.

Prediction goes through the shared `predict_observation` (the one rollout primitive GEPA, replay,
and eval all use), so the numbers here are directly comparable to a `wmh eval` run on the same step.
Metering is via a `MeteredProvider` so tokens/cost come from the real completion usage.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from wmh.core.render import render_action
from wmh.core.types import Trace
from wmh.optimize.gepa import predict_observation
from wmh.providers.base import Provider
from wmh.retrieval.leakfree import DemoRetriever
from wmh.tracking.clock import Clock, SystemClock
from wmh.tracking.metered import MeteredProvider
from wmh.tracking.tracker import Phase, RunTracker


class ScenarioStep(BaseModel):
    """One replayed step: the action, what the world model predicted, the recorded truth, and how
    long the prediction took (seconds, wall-clock for that single teacher-forced prediction)."""

    index: int
    action: str  # rendered action, for a human-readable scorecard
    predicted: str
    actual: str
    is_error_predicted: bool = False
    is_error_actual: bool = False
    seconds: float = 0.0


class ScenarioReport(BaseModel):
    """The outcome of replaying one scenario: per-step records plus wall-clock + cost totals.

    `startup_seconds` is the world model's cost-to-first-observation — the analogue of the real
    environment's container build — which for the world model is just one LLM round-trip (the first
    prediction), so it equals `steps[0].seconds`. `total_seconds` is the sum across all steps.
    `tokens`/`cost_usd` are the metered LLM spend to reconstruct the whole scenario. The real
    environment (run separately, closed-loop) is the comparison.
    """

    benchmark: str = ""
    model: str = ""
    trace_id: str = ""
    task: str | None = None
    steps: list[ScenarioStep] = Field(default_factory=list)
    startup_seconds: float = 0.0
    total_seconds: float = 0.0
    tokens: int = 0  # total LLM tokens metered across the scenario's steps
    cost_usd: float = 0.0  # total LLM cost for the scenario

    @property
    def fidelity(self) -> float:
        """Fraction of steps whose predicted error flag matched the recorded one (0..1).

        A cheap, judge-free fidelity signal for the at-a-glance comparison — the same ✓/≈ the live
        view shows. Rigorous scoring is `wmh bench run` (the rubric judge).
        """
        if not self.steps:
            return 0.0
        matches = sum(1 for s in self.steps if s.is_error_predicted == s.is_error_actual)
        return matches / len(self.steps)

    def summary(self) -> str:
        return (
            f"first observation in {self.startup_seconds:.2f}s, "
            f"{len(self.steps)} steps in {self.total_seconds:.2f}s total, "
            f"{self.tokens} tokens, ${self.cost_usd:.4f}, fidelity {self.fidelity:.0%}"
        )


def run_scenario(
    provider: Provider,
    env_prompt: str,
    trace: Trace,
    demos: DemoRetriever,
    *,
    benchmark: str = "",
    model: str = "",
    clock: Clock | None = None,
    on_step: Callable[[ScenarioStep], None] | None = None,
) -> ScenarioReport:
    """Open-loop teacher-forced replay of `trace` through `provider`/`env_prompt`, timed + metered.

    For each recorded step, predict the observation from the step's RECORDED `state_before` + action
    and its leak-free `demos` (via the shared `predict_observation` — the exact eval predict path),
    timing the call and capturing predicted-vs-recorded. This is open-loop: a step never sees the
    model's own prior predictions, so the result matches what `wmh eval` would score for that step.

    `demos` is a `DemoRetriever` built over the TRAIN corpus (it excludes the query's own trace), so
    retrieval is identical to evaluation. `on_step` is invoked with each `ScenarioStep` as it
    completes, for live rendering. Tokens/cost are metered off the real completion usage.
    """
    the_clock = clock or SystemClock()
    tracker = RunTracker(run_id=trace.trace_id, kind="scenario")
    metered = MeteredProvider(provider, tracker, base_phase=Phase.SERVE)

    task = trace.steps[0].task if trace.steps else None
    steps: list[ScenarioStep] = []
    total = 0.0
    for i, recorded in enumerate(trace.steps):
        start = the_clock.monotonic()
        predicted = predict_observation(
            metered,
            env_prompt,
            recorded.task,
            recorded.state_before,
            recorded.action,
            demos=demos.demos_for(trace.trace_id, recorded),
            # Teacher-forced prefix: the recorded prior steps (each carries only the agent's action
            # + the real recorded observation, never agent reasoning), so each prediction sees the
            # earlier turns exactly as the real environment did.
            history=trace.steps[:i],
        )
        elapsed = the_clock.monotonic() - start
        total += elapsed
        scenario_step = ScenarioStep(
            index=i,
            action=render_action(recorded.action),
            predicted=predicted.content,
            actual=recorded.observation.content,
            is_error_predicted=predicted.is_error,
            is_error_actual=recorded.observation.is_error,
            seconds=elapsed,
        )
        steps.append(scenario_step)
        if on_step is not None:
            on_step(scenario_step)

    usage = tracker.record_summary()
    return ScenarioReport(
        benchmark=benchmark,
        model=model,
        trace_id=trace.trace_id,
        task=task,
        steps=steps,
        startup_seconds=steps[0].seconds if steps else 0.0,
        total_seconds=total,
        tokens=usage.total.total_tokens,
        cost_usd=usage.total.cost_usd,
    )


__all__ = ["ScenarioStep", "ScenarioReport", "run_scenario"]
