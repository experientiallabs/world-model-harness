"""Tests for the timed scenario replay (race) — fake provider + scripted clock, no network."""

from __future__ import annotations

from wmh.bench.scenario import ScenarioReport, run_scenario
from wmh.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmh.engine.world_model import WorldModel
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind, TokenUsage
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


class FakeProvider:
    """Returns a fixed env-observation JSON; records how many completions it served."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="opus")
        self.calls = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        self.calls += 1
        return Completion(text='{"output": "predicted obs", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


class FakeClock:
    """Scripted monotonic clock: each call returns the next tick (a step takes a known delta)."""

    def __init__(self, ticks: list[float]) -> None:
        self._ticks = ticks
        self._i = 0

    def monotonic(self) -> float:
        if self._i >= len(self._ticks):
            raise AssertionError(
                f"FakeClock exhausted after {len(self._ticks)} ticks; "
                "the test needs more (run_scenario calls monotonic twice per step)"
            )
        value = self._ticks[self._i]
        self._i += 1
        return value


def _world_model(provider: FakeProvider) -> WorldModel:
    retriever = EmbeddingRetriever(HashingEmbedder(dim=32))
    return WorldModel(provider, retriever, env_prompt="P", top_k=3)


def _trace(tid: str, n: int = 2) -> Trace:
    return Trace(
        trace_id=tid,
        steps=[
            Step(
                action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"i": i}),
                observation=Observation(content=f"real-{i}"),
                state_before=EnvState(structured={"loc": "shop"}),
                task="look up users",
            )
            for i in range(n)
        ],
    )


def test_race_times_each_step_and_captures_predictions() -> None:
    provider = FakeProvider()
    # Two steps; clock pairs (start, end) per step: deltas 0.5s then 1.5s.
    clock = FakeClock([10.0, 10.5, 20.0, 21.5])
    report = run_scenario(
        _world_model(provider), _trace("t", n=2), benchmark="b", model="m", clock=clock
    )

    assert isinstance(report, ScenarioReport)
    assert report.benchmark == "b" and report.model == "m" and report.trace_id == "t"
    assert provider.calls == 2  # one LLM call per recorded step
    assert [s.seconds for s in report.steps] == [0.5, 1.5]
    assert report.startup_seconds == 0.5  # cost to FIRST observation
    assert abs(report.total_seconds - 2.0) < 1e-9
    # Predicted comes from the world model; actual is the recorded ground truth.
    assert report.steps[0].predicted == "predicted obs"
    assert report.steps[0].actual == "real-0"
    assert report.steps[1].actual == "real-1"


def test_race_empty_trace_is_safe() -> None:
    report = run_scenario(_world_model(FakeProvider()), _trace("empty", n=0))
    assert report.steps == []
    assert report.startup_seconds == 0.0
    assert report.total_seconds == 0.0


def test_race_default_clock_runs_without_a_fake() -> None:
    # No clock injected -> SystemClock; just assert it completes and times are non-negative.
    report = run_scenario(_world_model(FakeProvider()), _trace("t", n=1))
    assert len(report.steps) == 1
    assert report.steps[0].seconds >= 0.0
    assert report.steps[0].predicted == "predicted obs"


class MeteredProvider(FakeProvider):
    """Returns token usage on a priced model, so the race report's tokens/cost are non-zero."""

    def __init__(self) -> None:
        super().__init__()
        self.config = ProviderConfig(
            kind=ProviderKind.BEDROCK, model="us.anthropic.claude-opus-4-8"
        )

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        self.calls += 1
        return Completion(
            text='{"output": "predicted obs", "is_error": false}',
            usage=TokenUsage(input_tokens=100, output_tokens=20),
        )


def test_race_reports_tokens_cost_and_fidelity() -> None:
    # Two steps, 120 tokens each at Opus 4.8 rates -> tokens + cost rolled up from session metering.
    report = run_scenario(_world_model(MeteredProvider()), _trace("t", n=2))
    assert report.tokens == 240  # (100 + 20) * 2 steps
    # 200 input @ $5/Mtok + 40 output @ $25/Mtok = $0.001 + $0.001 = $0.002.
    assert abs(report.cost_usd - 0.002) < 1e-9
    # The recorded observations are non-error and the fake predicts is_error=false -> 100% match.
    assert report.fidelity == 1.0
    assert "tokens" in report.summary() and "fidelity" in report.summary()


def test_race_fidelity_zero_for_empty_trace() -> None:
    report = run_scenario(_world_model(FakeProvider()), _trace("empty", n=0))
    assert report.fidelity == 0.0
    assert report.tokens == 0
    assert report.cost_usd == 0.0
