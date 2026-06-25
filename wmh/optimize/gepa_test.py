"""Tests for the GEPAOptimizer (drives the `gepa` engine via WorldModelGEPAAdapter).

A deterministic fake Provider + fake Judge stand in for the real models: the provider returns a
fixed env prediction and a fixed improved prompt on reflection; the judge returns a fixed score and
critique. We assert the optimizer runs a bounded loop and returns a valid frontier — no network.
"""

from __future__ import annotations

from wmh.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmh.optimize.gepa import (
    ENV_PROMPT_COMPONENT,
    GEPAOptimizer,
    Optimizer,
    OptimizeResult,
    WorldModelGEPAAdapter,
    predict_observation,
)
from wmh.optimize.judge import JudgeResult
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind


class FakeProvider:
    """Distinguishes reflection calls (system mentions improving the prompt) from rollouts."""

    def __init__(self, *, prediction: str = "predicted obs", mutation: str = "IMPROVED") -> None:
        self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")
        self._prediction = prediction
        self._mutation = mutation
        self.reflection_calls = 0
        self.rollout_calls = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        if "improve the system prompt" in system:
            self.reflection_calls += 1
            return Completion(text=self._mutation)
        self.rollout_calls += 1
        return Completion(text=self._prediction)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


class FakeJudge:
    """Constant score + critique; counts calls so we can assert the loop is bounded."""

    def __init__(self, score: float = 0.5) -> None:
        self._score = score
        self.calls = 0

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        self.calls += 1
        return JudgeResult(score=self._score, critique="add the item total to the response")


def _trace(tid: str, n: int = 2) -> Trace:
    steps = [
        Step(
            action=Action(kind=ActionKind.TOOL_CALL, name="f", arguments={"i": i}),
            observation=Observation(content=f"real-{i}"),
            state_before=EnvState(structured={"loc": "shop"}),
            task="check out",
        )
        for i in range(n)
    ]
    return Trace(trace_id=tid, steps=steps)


def test_predict_observation_uses_provider() -> None:
    provider = FakeProvider(prediction="the cart now has 1 item")
    obs = predict_observation(
        provider,
        "PROMPT",
        task="t",
        state=EnvState(),
        action=Action(kind=ActionKind.MESSAGE, content="hi"),
        demos=[],
    )
    assert obs.content == "the cart now has 1 item"


def test_optimizer_satisfies_protocol() -> None:
    assert isinstance(GEPAOptimizer(FakeProvider(), FakeJudge()), Optimizer)


def test_optimize_runs_bounded_loop_and_returns_valid_frontier() -> None:
    provider = FakeProvider()
    judge = FakeJudge(score=0.5)
    opt = GEPAOptimizer(provider, judge)
    budget = 12

    result = opt.optimize([_trace("tr1"), _trace("tr2")], [_trace("te1")], "BASE", budget)

    assert isinstance(result, OptimizeResult)
    assert result.prompt  # a non-empty winning prompt
    assert len(result.frontier) >= 1
    assert all(isinstance(p, str) and p for p in result.frontier)
    # The loop terminates and stays near the budget. GEPA treats max_metric_calls as a *soft* cap:
    # it finishes the in-flight iteration, so it can overshoot by up to a minibatch + valset eval.
    assert 0 < result.metrics.rollouts_used <= budget * 2
    assert 0.0 <= result.metrics.held_out_accuracy <= 1.0
    # The judge was actually consulted, and the loop terminated (didn't run forever).
    assert judge.calls > 0


def test_optimize_with_zero_budget_returns_base_prompt() -> None:
    result = GEPAOptimizer(FakeProvider(), FakeJudge()).optimize(
        [_trace("tr1")], [_trace("te1")], "BASE", budget=0
    )
    assert result.prompt == "BASE"
    assert result.frontier == ["BASE"]


def test_optimize_with_no_traces_returns_base_prompt() -> None:
    result = GEPAOptimizer(FakeProvider(), FakeJudge()).optimize([], [], "BASE", budget=10)
    assert result.prompt == "BASE"
    assert result.frontier == ["BASE"]


def test_adapter_evaluate_scores_and_captures_traces() -> None:
    adapter = WorldModelGEPAAdapter(FakeProvider(), FakeJudge(score=0.7))
    batch = _trace("t", n=2).steps
    out = adapter.evaluate(batch, {ENV_PROMPT_COMPONENT: "P"}, capture_traces=True)
    assert out.scores == [0.7, 0.7]
    assert out.trajectories is not None and len(out.trajectories) == 2

    reflective = adapter.make_reflective_dataset(
        {ENV_PROMPT_COMPONENT: "P"}, out, [ENV_PROMPT_COMPONENT]
    )
    records = reflective[ENV_PROMPT_COMPONENT]
    assert len(records) == 2
    assert "Feedback" in records[0] and "Generated Outputs" in records[0]


def test_adapter_evaluate_survives_rollout_failure() -> None:
    class BoomJudge(FakeJudge):
        def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
            raise RuntimeError("judge exploded")

    adapter = WorldModelGEPAAdapter(FakeProvider(), BoomJudge())
    out = adapter.evaluate(_trace("t", n=1).steps, {ENV_PROMPT_COMPONENT: "P"}, capture_traces=True)
    # Per-example failure -> fallback score, never an exception.
    assert out.scores == [0.0]
    assert out.trajectories is not None and "failed" in out.trajectories[0].critique
