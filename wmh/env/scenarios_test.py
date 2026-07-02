"""Tests for scenario extraction from traces."""

from __future__ import annotations

from wmh.core.types import Action, ActionKind, Observation, Step, Trace
from wmh.env.scenarios import Scenario, scenarios_from_traces


def _trace(trace_id: str, task: str | None) -> Trace:
    step = Step(
        action=Action(kind=ActionKind.TOOL_CALL, name="t", arguments={}),
        observation=Observation(content="ok"),
        task=task,
    )
    return Trace(trace_id=trace_id, steps=[step])


def test_extracts_unique_tasks_with_provenance() -> None:
    traces = [
        _trace("a", "book a flight"),
        _trace("b", "cancel order 7"),
        _trace("c", "book a flight"),  # duplicate task -> same scenario, extra provenance
    ]
    scenarios = scenarios_from_traces(traces)
    assert scenarios == [
        Scenario(task="book a flight", provenance=["a", "c"]),
        Scenario(task="cancel order 7", provenance=["b"]),
    ]


def test_skips_traces_without_a_task() -> None:
    traces = [_trace("a", None), _trace("b", "   "), _trace("c", "real task")]
    scenarios = scenarios_from_traces(traces)
    assert [s.task for s in scenarios] == ["real task"]


def test_empty_input_gives_empty_output() -> None:
    assert scenarios_from_traces([]) == []
