"""Tests for the FastAPI serving layer, with an injected in-process WorldModel (no network)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from wmh.core.types import Action, ActionKind, Observation, Step, Trace
from wmh.engine.world_model import WorldModel
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder
from wmh.serving.server import create_app


class FakeProvider:
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
        return Completion(text='{"output": "user found", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def _client() -> TestClient:
    retriever = EmbeddingRetriever(HashingEmbedder(dim=32))
    retriever.index(
        [
            Trace(
                trace_id="t",
                steps=[
                    Step(
                        action=Action(
                            kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "u1"}
                        ),
                        observation=Observation(content="found u1"),
                    )
                ],
            )
        ]
    )
    wm = WorldModel(FakeProvider(), retriever, top_k=3)
    return TestClient(create_app(world_model=wm))


def test_healthz() -> None:
    assert _client().get("/healthz").json() == {"status": "ok"}


def test_session_lifecycle_and_step() -> None:
    client = _client()
    resp = client.post("/sessions", json={"task": "look up a user"})
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    step = client.post(
        f"/sessions/{session_id}/step",
        json={"action": {"kind": "tool_call", "name": "get_user", "arguments": {"id": "u2"}}},
    )
    assert step.status_code == 200
    assert step.json()["observation"]["content"] == "user found"

    got = client.get(f"/sessions/{session_id}")
    assert got.status_code == 200
    assert len(got.json()["history"]) == 1


def test_step_on_missing_session_is_404() -> None:
    client = _client()
    resp = client.post(
        "/sessions/nope/step",
        json={"action": {"kind": "message", "content": "hi"}},
    )
    assert resp.status_code == 404
