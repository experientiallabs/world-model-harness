"""Tests for the replay/reconstruction-fidelity harness, with fakes (no network)."""

from __future__ import annotations

from wmh.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmh.engine.replay import replay
from wmh.optimize.judge import JudgeResult
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


class FakeProvider:
    def __init__(self, reply: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._reply = reply
        self.last_user: str | None = None

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        self.last_user = messages[0].content
        return Completion(text=self._reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


class FakeJudge:
    def __init__(self, score: float) -> None:
        self._score = score

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        return JudgeResult(score=self._score, critique="ok")


def _trace(tid: str, n: int = 2) -> Trace:
    return Trace(
        trace_id=tid,
        steps=[
            Step(
                action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"i": i}),
                observation=Observation(content=f"real-{i}", is_error=False),
                state_before=EnvState(structured={"loc": "shop"}),
                task="look up",
            )
            for i in range(n)
        ],
    )


def test_replay_scores_and_aggregates() -> None:
    provider = FakeProvider('{"output": "real-0", "is_error": false}')
    report = replay("BASE", [_trace("h", n=2)], provider, FakeJudge(0.8))
    assert report.n_steps == 2
    assert report.mean_score == 0.8
    # Predicted is_error (false) matches actual (false) for both.
    assert report.error_flag_accuracy == 1.0
    assert report.results[0].actual == "real-0"


def test_replay_tracks_error_flag_mismatch() -> None:
    # Model predicts an error, but the actual observation is not an error -> flag mismatch.
    provider = FakeProvider('{"output": "boom", "is_error": true}')
    report = replay("BASE", [_trace("h", n=1)], provider, FakeJudge(0.0))
    assert report.error_flag_accuracy == 0.0
    assert report.results[0].is_error_predicted is True
    assert report.results[0].is_error_actual is False


def test_replay_rag_is_leakfree() -> None:
    # The held-out trace's own steps must never appear as demos in its prompt.
    train = [_trace("train-A", n=2)]
    holdout = [_trace("train-A", n=2)]  # same trace_id as a train trace -> must be excluded
    provider = FakeProvider('{"output": "x", "is_error": false}')
    retriever = EmbeddingRetriever(HashingEmbedder(dim=64))
    report = replay(
        "BASE", holdout, provider, FakeJudge(0.5), retriever=retriever, train=train, top_k=3
    )
    assert report.n_steps == 2
    # With train and holdout sharing the trace_id, every demo is excluded -> no leakage into prompt.
    assert "real-" not in (provider.last_user or "").split("SIMILAR PAST EXAMPLES")[-1]


def test_replay_empty_is_safe() -> None:
    report = replay("BASE", [], FakeProvider("{}"), FakeJudge(1.0))
    assert report.n_steps == 0
    assert report.mean_score == 0.0
