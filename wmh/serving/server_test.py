"""Tests for the FastAPI serving layer, with injected in-process WorldModels (no network)."""

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


def _world_model() -> WorldModel:
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
    return WorldModel(FakeProvider(), retriever, top_k=3)


def _client(world_models: dict[str, WorldModel] | None = None) -> TestClient:
    models = world_models or {"airline": _world_model()}
    return TestClient(create_app(world_models=models))


def test_healthz() -> None:
    assert _client().get("/healthz").json() == {"status": "ok"}


def test_lists_world_models_by_name() -> None:
    client = _client({"airline": _world_model(), "retail": _world_model()})
    assert client.get("/world_models").json() == {"world_models": ["airline", "retail"]}


def test_session_lifecycle_and_step_are_namespaced() -> None:
    client = _client()
    resp = client.post("/world_models/airline/sessions", json={"task": "look up a user"})
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    step = client.post(
        f"/world_models/airline/sessions/{session_id}/step",
        json={"action": {"kind": "tool_call", "name": "get_user", "arguments": {"id": "u2"}}},
    )
    assert step.status_code == 200
    assert step.json()["observation"]["content"] == "user found"

    got = client.get(f"/world_models/airline/sessions/{session_id}")
    assert got.status_code == 200
    assert len(got.json()["history"]) == 1


def test_unknown_world_model_is_404() -> None:
    client = _client()
    resp = client.post("/world_models/nope/sessions", json={"task": "x"})
    assert resp.status_code == 404


def test_step_on_missing_session_is_404() -> None:
    client = _client()
    resp = client.post(
        "/world_models/airline/sessions/nope/step",
        json={"action": {"kind": "message", "content": "hi"}},
    )
    assert resp.status_code == 404


def test_sessions_are_isolated_between_named_models() -> None:
    client = _client({"airline": _world_model(), "retail": _world_model()})
    created = client.post("/world_models/airline/sessions", json={"task": "x"})
    session_id = created.json()["session_id"]
    # A session created on `airline` is not visible under `retail`.
    miss = client.get(f"/world_models/retail/sessions/{session_id}")
    assert miss.status_code == 404
