"""Tests for the train-vs-eval temperature ablation (drives the framework with fakes)."""

from __future__ import annotations

from wmh.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmh.optimize.judge import JudgeResult
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.research.ablation import Ablation, run_ablation
from wmh.research.temperature import (
    EVAL_TEMP,
    TRAIN_TEMP,
    TemperatureAblation,
    temperature_conditions,
)


class FakeProvider:
    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")
        self.eval_temps: list[float] = []
        self._optimizing = False

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        if "improve the system prompt" in system:
            return Completion(text="IMPROVED")
        return Completion(text="predicted")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


class FakeJudge:
    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        return JudgeResult(score=0.5, critique="c")


def _trace(tid: str, n: int = 2) -> Trace:
    return Trace(
        trace_id=tid,
        steps=[
            Step(
                action=Action(kind=ActionKind.TOOL_CALL, name="f", arguments={"i": i}),
                observation=Observation(content=f"real-{i}"),
                state_before=EnvState(structured={"loc": "shop"}),
                task="t",
            )
            for i in range(n)
        ],
    )


def _ablation() -> TemperatureAblation:
    return TemperatureAblation(
        [_trace("tr1"), _trace("tr2")],
        [_trace("te1")],
        "BASE",
        make_backends=lambda: (FakeProvider(), FakeJudge(), None),
        budget=6,
    )


def test_default_grid_is_2x2_crossed() -> None:
    conds = temperature_conditions()
    assert len(conds) == 4
    labels = {c.label for c in conds}
    assert labels == {
        "Ttrain=0/Teval=0",
        "Ttrain=0/Teval=1",
        "Ttrain=1/Teval=0",
        "Ttrain=1/Teval=1",
    }
    # Each condition carries both knobs.
    for c in conds:
        assert TRAIN_TEMP in c.params and EVAL_TEMP in c.params


def test_temperature_ablation_satisfies_protocol() -> None:
    assert isinstance(_ablation(), Ablation)


def test_run_one_condition_returns_holdout_fidelity() -> None:
    ablation = _ablation()
    cond = ablation.conditions()[0]
    score = ablation.run(cond, seed=0)
    # The fake judge returns 0.5 for every held-out step, so the mean fidelity is 0.5.
    assert abs(score - 0.5) < 1e-9


def test_run_ablation_aggregates_all_cells_across_seeds() -> None:
    ablation = _ablation()
    report = run_ablation(ablation, [0, 1])
    assert report.name == "train-vs-eval-temperature"
    assert len(report.conditions) == 4
    for cell in report.conditions:
        assert len(cell.per_seed) == 2
        # Deterministic fakes -> identical scores across seeds -> zero std.
        assert abs(cell.mean - 0.5) < 1e-9
        assert cell.std == 0.0
