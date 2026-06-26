"""Tests for the demo round-trip, with a fake agent + world-model provider (no network)."""

from __future__ import annotations

from wmh.core.types import Action, ActionKind, Observation, Step, Trace
from wmh.engine.demo import run_demo
from wmh.engine.world_model import WorldModel
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


class ScriptedProvider:
    """Returns an agent tool-call JSON first, then world-model JSON for the env step."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        if "role-play an agent" in system:
            return Completion(text='{"name": "get_user", "arguments": {"id": "u1"}}')
        return Completion(text='{"output": "found u1", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def test_run_demo_produces_action_prompt_and_observation() -> None:
    provider = ScriptedProvider()
    retriever = EmbeddingRetriever(HashingEmbedder(dim=32))
    examples = [
        Step(
            action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "u0"}),
            observation=Observation(content="found u0"),
            task="look up users",
        )
    ]
    retriever.index([Trace(trace_id="t", steps=examples)])
    wm = WorldModel(provider, retriever, top_k=3)

    result = run_demo(wm, provider, examples)
    assert result.agent_action.kind == ActionKind.TOOL_CALL
    assert result.agent_action.name == "get_user"
    assert "get_user" in result.env_prompt  # the retrieved demo shows up in the prompt
    assert result.observation.content == "found u1"
