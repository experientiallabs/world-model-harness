"""Tests for the WorldModel session lifecycle."""

from __future__ import annotations

from wmh.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmh.engine.world_model import WorldModel
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


def test_world_model_new_session_works() -> None:
    wm = WorldModel.__new__(WorldModel)
    wm._sessions = {}
    session = WorldModel.new_session(wm, task="hi")
    assert session.id
    assert WorldModel.get_session(wm, session.id) is session


class FakeProvider:
    """Returns a canned world-model JSON completion; captures the last prompt for assertions."""

    def __init__(self, reply: str) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="m")
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


def _retriever_with(steps: list[Step]) -> EmbeddingRetriever:
    r = EmbeddingRetriever(HashingEmbedder(dim=64))
    r.index([Trace(trace_id="t", steps=steps)])
    return r


def test_step_predicts_parses_and_advances_session() -> None:
    provider = FakeProvider(
        '{"output": "user found: alice", "is_error": false, "state_note": "looked up alice"}'
    )
    demo = Step(
        action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "bob"}),
        observation=Observation(content="user found: bob"),
    )
    wm = WorldModel(provider, _retriever_with([demo]), top_k=3)
    session = wm.new_session(task="look up alice")

    obs = wm.step(
        session.id, Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "alice"})
    )

    assert obs.content == "user found: alice"
    assert obs.is_error is False
    # The retrieved demo made it into the prompt.
    assert provider.last_user is not None and "get_user" in provider.last_user
    # Session advanced: history grew and the scratchpad recorded the state note.
    assert len(session.history) == 1
    assert "looked up alice" in session.state.scratchpad


def test_step_marks_errors_and_enriches_buffer() -> None:
    provider = FakeProvider('{"output": "no such reservation", "is_error": true}')
    retriever = _retriever_with([])
    wm = WorldModel(provider, retriever, top_k=3)
    session = wm.new_session(task="check r_999")

    obs = wm.step(
        session.id,
        Action(kind=ActionKind.TOOL_CALL, name="get_reservation", arguments={"id": "r_999"}),
    )
    assert obs.is_error is True
    # The freshly produced step was added to the buffer (online enrichment).
    assert len(retriever._steps) == 1


def test_load_reads_artifact(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    from wmh.config import ArtifactPaths, HarnessConfig, save_config

    root = tmp_path / ".wmh"
    # embed_dim must match the embedder the index was built with (64 here), or load() rebuilds a
    # mismatched query embedder. This is the contract WorldModel.load relies on.
    save_config(HarnessConfig(top_k=2, embed_dim=64), root)
    paths = ArtifactPaths(root)
    paths.optimized_prompt.parent.mkdir(parents=True, exist_ok=True)
    paths.optimized_prompt.write_text("OPTIMIZED ENV PROMPT", encoding="utf-8")
    r = _retriever_with(
        [
            Step(
                action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "x"}),
                observation=Observation(content="ok"),
            )
        ]
    )
    r.save(paths.index)

    wm = WorldModel.load(str(root), FakeProvider("{}"))
    assert wm._env_prompt == "OPTIMIZED ENV PROMPT"
    assert wm._top_k == 2
    # The persisted index was reloaded: the stored step is retrievable.
    restored = wm._retriever.topk(
        EnvState(), Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={"id": "x"}), k=1
    )
    assert len(restored) == 1 and restored[0].observation.content == "ok"
