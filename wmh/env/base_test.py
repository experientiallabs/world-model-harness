"""Tests for the Env protocol and the WorldModelEnv backend."""

from __future__ import annotations

import pytest

from wmh.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmh.engine.world_model import WorldModel
from wmh.env import Env, WorldModelEnv
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


class FakeProvider:
    """Returns a canned world-model JSON completion."""

    def __init__(self, reply: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
        self._reply = reply

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        return Completion(text=self._reply)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def _world_model(reply: str) -> WorldModel:
    demo = Step(
        action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "bob"}),
        observation=Observation(content="user found: bob"),
    )
    retriever = EmbeddingRetriever(HashingEmbedder(dim=64))
    retriever.index([Trace(trace_id="t", steps=[demo])])
    return WorldModel(FakeProvider(reply), retriever, top_k=1)


def test_world_model_env_satisfies_protocol() -> None:
    env = WorldModelEnv(_world_model('{"output": "ok", "is_error": false}'))
    assert isinstance(env, Env)


def test_world_model_env_episode_lifecycle() -> None:
    wm = _world_model('{"output": "user found: alice", "is_error": false}')
    env = WorldModelEnv(wm)

    state = env.reset(task="look up alice", seed_state=EnvState(scratchpad="fresh"))
    assert state.scratchpad == "fresh"
    assert wm.get_session(env.session_id).task == "look up alice"

    obs = env.step(Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "alice"}))
    assert obs.content == "user found: alice"
    assert len(wm.get_session(env.session_id).history) == 1

    env.close()
    with pytest.raises(RuntimeError, match="call reset"):
        _ = env.session_id


def test_world_model_env_reset_starts_fresh_session() -> None:
    env = WorldModelEnv(_world_model('{"output": "ok", "is_error": false}'))
    env.reset(task="a")
    first = env.session_id
    env.reset(task="b")
    assert env.session_id != first
