"""Timed open-loop scenario replay through a live world model.

Where `wmh.bench.runner` *scores fidelity* over many seeds, this *plays one recorded scenario* and
**times it**: it replays a recorded trace's `(state, action)` steps through the real serving path
(`WorldModel.step`) in order, measuring how long each predicted observation takes and the LLM cost,
and comparing each prediction to the recorded ground truth. It is the world-model half of a
scenario comparison — `wmh bench scenario <name>` reconstructs the environment with the LLM (no
container to boot); the real environment runs the same recorded commands separately (see the
`tools/<benchmark>-capture/` runners) and you compare the two end times.

This module owns no I/O and no LLM specifics: it takes an already-loaded `WorldModel`, an
already-ingested `Trace`, and an injectable `Clock`, so it is unit-testable with a fake provider and
a scripted clock. The CLI (`wmh bench scenario`) does the loading and rendering.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from wmh.core.render import render_action
from wmh.core.types import Trace
from wmh.engine.world_model import WorldModel
from wmh.tracking.clock import Clock, SystemClock


class ScenarioStep(BaseModel):
    """One replayed step: the action, what the world model predicted, the recorded truth, and how
    long the prediction took (seconds, wall-clock for that single `step` call)."""

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
    environment's container boot — which for the world model is just one LLM round-trip (the first
    `step`), so it equals `steps[0].seconds`. `total_seconds` is the sum across all steps.
    `tokens`/`cost_usd` come from the world model's serve-time metering (`session_usage`) — the LLM
    spend to reconstruct the whole scenario. These are the numbers `wmh bench scenario` prints; the
    real environment (run separately) is the comparison.
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
    world_model: WorldModel,
    trace: Trace,
    *,
    benchmark: str = "",
    model: str = "",
    clock: Clock | None = None,
    on_step: Callable[[ScenarioStep], None] | None = None,
) -> ScenarioReport:
    """Replay `trace`'s recorded steps through `world_model`, timing each prediction.

    Seeds a session from the trace's task and the first step's `state_before`, then steps each
    recorded action in order through the live serving path. Each step is timed with `clock`
    (`SystemClock` by default; a fake clock makes the timing deterministic in tests). The predicted
    observation is captured alongside the recorded one so the comparison can show them converging.

    `on_step` is invoked with each `ScenarioStep` as it completes, so a caller (the CLI) can render
    observations live as the world model fills the scenario in.
    """
    the_clock = clock or SystemClock()
    # The task is recorded per-step (the originating instruction); take it from the first step.
    task = trace.steps[0].task if trace.steps else None
    seed_state = trace.steps[0].state_before if trace.steps else None
    session = world_model.new_session(task=task, seed_state=seed_state)

    steps: list[ScenarioStep] = []
    total = 0.0
    for i, recorded in enumerate(trace.steps):
        start = the_clock.monotonic()
        predicted = world_model.step(session.id, recorded.action)
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

    # Serve-time metering: the world model meters every `step`'s LLM tokens/cost onto the session's
    # tracker, so read the scenario's total spend back from `session_usage` (no extra LLM calls).
    usage = world_model.session_usage(session.id)
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
