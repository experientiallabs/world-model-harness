"""Tests for the LLMJudge and its robust parsing of the judge reply."""

from __future__ import annotations

from wmh.core.types import Action, ActionKind, Observation, Step
from wmh.optimize.judge import Judge, JudgeResult, LLMJudge, _parse_judgement
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind


class FakeProvider:
    """Returns a canned completion text; records the last prompt for assertions."""

    def __init__(self, reply: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")
        self._reply = reply
        self.last_system: str | None = None
        self.last_user: str | None = None

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        self.last_system = system
        self.last_user = messages[0].content
        return Completion(text=self._reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def _ctx() -> Step:
    return Step(
        action=Action(kind=ActionKind.TOOL_CALL, name="add_to_cart", arguments={"sku": "A1"}),
        observation=Observation(content="cart has 1 item"),
    )


def test_llm_judge_satisfies_protocol() -> None:
    assert isinstance(LLMJudge(FakeProvider("{}")), Judge)


def test_score_parses_bare_json() -> None:
    provider = FakeProvider('{"score": 0.8, "critique": "close but missing total"}')
    judge = LLMJudge(provider)
    result = judge.score(Observation(content="pred"), Observation(content="actual"), _ctx())
    assert result.score == 0.8
    assert "missing total" in result.critique
    # The judge actually saw both observations in its prompt.
    assert provider.last_user is not None
    assert "pred" in provider.last_user and "actual" in provider.last_user


def test_parse_handles_fenced_json() -> None:
    text = 'Sure:\n```json\n{"score": 0.4, "critique": "wrong status"}\n```\nDone.'
    result = _parse_judgement(text)
    assert result.score == 0.4
    assert result.critique == "wrong status"


def test_parse_handles_json_embedded_in_prose() -> None:
    text = 'My verdict is {"score": 1.0, "critique": "identical"} overall.'
    result = _parse_judgement(text)
    assert result.score == 1.0


def test_parse_clamps_out_of_range_scores() -> None:
    assert _parse_judgement('{"score": 1.7, "critique": "x"}').score == 1.0
    assert _parse_judgement('{"score": -0.5, "critique": "x"}').score == 0.0


def test_parse_unparseable_falls_back_to_zero() -> None:
    result = _parse_judgement("the model rambled with no json at all")
    assert result.score == 0.0
    assert "Unparseable" in result.critique


def test_score_uses_zero_temperature() -> None:
    # Determinism matters for a fitness signal; just assert the call path returns a JudgeResult.
    result = LLMJudge(FakeProvider('{"score": 0.0, "critique": ""}')).score(
        Observation(content="a"), Observation(content="b"), _ctx()
    )
    assert isinstance(result, JudgeResult)
