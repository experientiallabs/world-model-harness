"""Timed scenario replay against a live world model — the "world model" side of the sandbox race.

Where `wmh.bench.runner` *scores fidelity* over many seeds, this *plays one recorded scenario* and
**times it**: it replays a recorded trace's `(state, action)` steps through the real serving path
(`WorldModel.step`) in order, measuring how long each predicted observation takes and comparing it
to the recorded ground truth. The point is the side-by-side demo (`docs/...`): a real sandbox pays a
large startup cost before its first step, while the world model — already loaded, no container to
boot — answers the first action after a single LLM round-trip.

This module owns no I/O and no LLM specifics: it takes an already-loaded `WorldModel`, an
already-ingested `Trace`, and an injectable `Clock`, so it is unit-testable with a fake provider and
a scripted clock. The CLI (`wmh bench race`) does the loading and rendering.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from wmh.core.render import render_action
from wmh.core.types import Trace
from wmh.engine.world_model import WorldModel
from wmh.tracking.clock import Clock, SystemClock


class RaceStep(BaseModel):
    """One replayed step: the action, what the world model predicted, the recorded truth, and how
    long the prediction took (seconds, wall-clock for that single `step` call)."""

    index: int
    action: str  # rendered action, for a human-readable scorecard
    predicted: str
    actual: str
    is_error_predicted: bool = False
    is_error_actual: bool = False
    seconds: float = 0.0


class RaceReport(BaseModel):
    """The outcome of replaying one scenario: per-step records plus wall-clock + cost totals.

    `startup_seconds` is the world model's cost-to-first-observation — the analogue of the sandbox's
    container boot — which for the world model is just one LLM round-trip (the first `step`), so it
    equals `steps[0].seconds`. `total_seconds` is the sum across all steps. `tokens`/`cost_usd` come
    from the world model's serve-time metering (`session_usage`) — the LLM spend to reconstruct the
    whole scenario. These are the numbers the demo's right-side panel shows (time + cost); the
    sandbox side (booted separately) is the comparison.
    """

    benchmark: str = ""
    model: str = ""
    trace_id: str = ""
    task: str | None = None
    steps: list[RaceStep] = Field(default_factory=list)
    startup_seconds: float = 0.0
    total_seconds: float = 0.0
    tokens: int = 0  # total LLM tokens metered across the scenario's steps
    cost_usd: float = 0.0  # total LLM cost for the scenario

    @property
    def fidelity(self) -> float:
        """Fraction of steps whose predicted error flag matched the recorded one (0..1).

        A cheap, judge-free fidelity signal for the demo — the same ✓/≈ the live view shows. Real
        scoring is `wmh bench run` (the rubric judge); this is the at-a-glance number for the race.
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


def race_trace(
    world_model: WorldModel,
    trace: Trace,
    *,
    benchmark: str = "",
    model: str = "",
    clock: Clock | None = None,
    on_step: Callable[[RaceStep], None] | None = None,
) -> RaceReport:
    """Replay `trace`'s recorded steps through `world_model`, timing each prediction.

    Seeds a session from the trace's task and the first step's `state_before`, then steps each
    recorded action in order through the live serving path. Each step is timed with `clock`
    (`SystemClock` by default; a fake clock makes the timing deterministic in tests). The predicted
    observation is captured alongside the recorded one so the demo can show them converging.

    `on_step` is invoked with each `RaceStep` as it completes, so a caller (the CLI) can render
    observations live — the demo's right side filling in while the sandbox is still booting.
    """
    the_clock = clock or SystemClock()
    # The task is recorded per-step (the originating instruction); take it from the first step.
    task = trace.steps[0].task if trace.steps else None
    seed_state = trace.steps[0].state_before if trace.steps else None
    session = world_model.new_session(task=task, seed_state=seed_state)

    steps: list[RaceStep] = []
    total = 0.0
    for i, recorded in enumerate(trace.steps):
        start = the_clock.monotonic()
        predicted = world_model.step(session.id, recorded.action)
        elapsed = the_clock.monotonic() - start
        total += elapsed
        race_step = RaceStep(
            index=i,
            action=render_action(recorded.action),
            predicted=predicted.content,
            actual=recorded.observation.content,
            is_error_predicted=predicted.is_error,
            is_error_actual=recorded.observation.is_error,
            seconds=elapsed,
        )
        steps.append(race_step)
        if on_step is not None:
            on_step(race_step)

    # Serve-time metering: the world model meters every `step`'s LLM tokens/cost onto the session's
    # tracker, so read the scenario's total spend back from `session_usage` (no extra LLM calls).
    usage = world_model.session_usage(session.id)
    return RaceReport(
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


__all__ = ["RaceStep", "RaceReport", "race_trace"]
