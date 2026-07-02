"""Tests for the WorldModel session lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import wmh.telemetry as telemetry
from wmh.config import ArtifactPaths, HarnessConfig, save_config
from wmh.config.settings import set_telemetry_enabled
from wmh.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmh.engine.grounding import GroundingResult
from wmh.engine.knowledge import KnowledgeBase
from wmh.engine.world_model import WorldModel
from wmh.providers.base import (
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder


def test_world_model_new_session_works() -> None:
    wm = WorldModel.__new__(WorldModel)
    wm._telemetry_root = Path(".wmh")
    wm._sessions = {}
    wm._trackers = {}
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
        max_tokens: int = 8192,
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


class _UsageProvider(FakeProvider):
    """FakeProvider that also reports token usage, for serve-metering assertions."""

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.last_system = system
        self.last_user = messages[0].content
        return Completion(text=self._reply, usage=TokenUsage(input_tokens=120, output_tokens=30))


def test_step_meters_usage_per_session() -> None:
    provider = _UsageProvider('{"output": "ok", "is_error": false}')
    provider.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="claude-opus-4-8")
    wm = WorldModel(provider, _retriever_with([]), top_k=1)
    session = wm.new_session(task="t")

    wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="f", arguments={}))
    wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="f", arguments={}))

    usage = wm.session_usage(session.id)
    assert usage.kind == "serve"
    assert usage.total.calls == 2
    assert usage.total.input_tokens == 240
    assert usage.total.output_tokens == 60
    # 240*5/1e6 + 60*25/1e6 = 0.0012 + 0.0015 = 0.0027 (float division → approx)
    assert usage.total.cost_usd == pytest.approx(0.0027)


class _SequenceProvider(FakeProvider):
    """Returns queued replies in order (for the grounding two-completion flow)."""

    def __init__(self, replies: list[str]) -> None:
        super().__init__(replies[0])
        self._replies = list(replies)
        self.users: list[str] = []

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 8192,
    ) -> Completion:
        self.last_system = system
        self.last_user = messages[0].content
        self.users.append(messages[0].content)
        return Completion(text=self._replies.pop(0))


class _RecordingGrounder:
    """Returns one canned hit and records the queries it served."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def ground(self, query: str) -> list[GroundingResult]:
        self.queries.append(query)
        return [GroundingResult(title="hit", url="https://x", snippet=f"facts about {query}")]


def _kb(tmp_path: Path) -> KnowledgeBase:
    kb = KnowledgeBase(tmp_path / "knowledge")
    kb.write_file("rules.md", "- gate: modifying a booking requires auth")
    return kb


def test_step_defaults_render_no_knowledge_and_base_contract() -> None:
    provider = FakeProvider('{"output": "ok", "is_error": false}')
    wm = WorldModel(provider, _retriever_with([]), top_k=1)
    session = wm.new_session(task="t")
    wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="f", arguments={}))
    user = provider.last_user or ""
    assert "KNOWLEDGE BASE" not in user
    assert '"reasoning"' not in user


def test_step_renders_knowledge_and_reasoning_contract(tmp_path: Path) -> None:
    provider = FakeProvider('{"reasoning": "auth ok", "output": "done", "is_error": false}')
    wm = WorldModel(provider, _retriever_with([]), top_k=1, knowledge=_kb(tmp_path), reasoning=True)
    session = wm.new_session(task="t")
    obs = wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="f", arguments={}))
    user = provider.last_user or ""
    assert "gate: modifying a booking requires auth" in user
    assert '"reasoning"' in user
    assert obs.content == "done"  # deliberation never reaches the agent


def test_step_appends_kb_note_to_learned(tmp_path: Path) -> None:
    provider = FakeProvider(
        '{"reasoning": "r", "output": "ok", "is_error": false, '
        '"kb_note": "flight HAT-201 JFK->SFO exists"}'
    )
    kb = _kb(tmp_path)
    wm = WorldModel(provider, _retriever_with([]), top_k=1, knowledge=kb, reasoning=True)
    session = wm.new_session(task="t")
    wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="f", arguments={}))
    learned = (tmp_path / "knowledge" / "learned.md").read_text(encoding="utf-8")
    assert "flight HAT-201 JFK->SFO exists" in learned
    assert session.id[:8] in learned  # provenance
    # Seeded files were not touched.
    assert (tmp_path / "knowledge" / "rules.md").read_text(
        encoding="utf-8"
    ) == "- gate: modifying a booking requires auth"


def test_step_ground_query_searches_recompletes_and_caches(tmp_path: Path) -> None:
    provider = _SequenceProvider(
        [
            '{"reasoning": "unknown pkg", "output": "", "is_error": false, '
            '"ground_query": "tomli_w python package"}',
            '{"reasoning": "grounded", "output": "tomli_w 1.0.0 installed", "is_error": false}',
        ]
    )
    grounder = _RecordingGrounder()
    kb = _kb(tmp_path)
    wm = WorldModel(
        provider, _retriever_with([]), top_k=1, knowledge=kb, reasoning=True, grounder=grounder
    )
    session = wm.new_session(task="t")
    obs = wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="pip", arguments={}))

    assert grounder.queries == ["tomli_w python package"]
    assert obs.content == "tomli_w 1.0.0 installed"  # the re-completion's observation wins
    # Second completion saw the search results.
    assert "facts about tomli_w python package" in provider.users[1]
    # Results were cached into the KB, so the same entity is never searched twice.
    assert kb.lookup_grounded("tomli_w python package") is not None


def test_step_ground_query_cache_hit_skips_search(tmp_path: Path) -> None:
    kb = _kb(tmp_path)
    kb.append_grounded("tomli_w python package", "- cached: tomli_w writes TOML")
    provider = _SequenceProvider(
        [
            '{"reasoning": "unknown", "output": "", "is_error": false, '
            '"ground_query": "tomli_w python package"}',
            '{"reasoning": "grounded", "output": "ok", "is_error": false}',
        ]
    )
    grounder = _RecordingGrounder()
    wm = WorldModel(
        provider, _retriever_with([]), top_k=1, knowledge=kb, reasoning=True, grounder=grounder
    )
    session = wm.new_session(task="t")
    wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="pip", arguments={}))
    assert grounder.queries == []  # served from grounded.md
    assert "cached: tomli_w writes TOML" in provider.users[1]


def test_step_grounding_budget_bounds_searches_per_session(tmp_path: Path) -> None:
    replies: list[str] = []
    for i in range(3):
        replies.append(
            f'{{"reasoning": "?", "output": "", "is_error": false, "ground_query": "entity {i}"}}'
        )
        replies.append('{"reasoning": "ok", "output": "obs", "is_error": false}')
    provider = _SequenceProvider(replies)
    grounder = _RecordingGrounder()
    wm = WorldModel(
        provider,
        _retriever_with([]),
        top_k=1,
        knowledge=_kb(tmp_path),
        reasoning=True,
        grounder=grounder,
        ground_budget=2,
    )
    session = wm.new_session(task="t")
    for _ in range(3):
        wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="f", arguments={}))
    # Third step's query was over budget: no search, no re-completion (5 provider calls, not 6).
    assert len(grounder.queries) == 2
    assert len(provider.users) == 5


def test_load_without_knowledge_dir_serves_unchanged(tmp_path: Path) -> None:
    root = tmp_path / ".wmh"
    save_config(HarnessConfig(serve_provider=ProviderKind.BEDROCK), root=root)
    provider = FakeProvider('{"output": "ok", "is_error": false}')
    wm = WorldModel.load(str(root), provider)
    session = wm.new_session(task="t")
    wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="f", arguments={}))
    assert "KNOWLEDGE BASE" not in (provider.last_user or "")


def test_load_picks_up_knowledge_dir_and_flags(tmp_path: Path) -> None:
    root = tmp_path / ".wmh"
    save_config(HarnessConfig(serve_provider=ProviderKind.BEDROCK, reasoning=True), root=root)
    KnowledgeBase(ArtifactPaths(root).knowledge).write_file("rules.md", "- gate: auth first")
    provider = FakeProvider('{"reasoning": "r", "output": "ok", "is_error": false}')
    wm = WorldModel.load(str(root), provider)
    session = wm.new_session(task="t")
    wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="f", arguments={}))
    user = provider.last_user or ""
    assert "gate: auth first" in user
    assert '"reasoning"' in user


def test_step_telemetry_counts_steps_without_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[object] = []

    class FakePosthog:
        def __init__(self, project_api_key: str, **kwargs: object) -> None:
            pass

        def capture(self, event: str, **kwargs: object) -> str:
            calls.append({"event": event, **kwargs})
            return "message-id"

        def shutdown(self) -> None:
            pass

    telemetry._CLIENTS.clear()
    monkeypatch.setattr(telemetry, "Posthog", FakePosthog)
    monkeypatch.setenv("WMH_TELEMETRY", "1")
    monkeypatch.setenv("WMH_POSTHOG_PROJECT_API_KEY", "phc_test")

    provider = _UsageProvider('{"output": "secret observation", "is_error": false}')
    provider.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="claude-opus-4-8")
    wm = WorldModel(provider, _retriever_with([]), top_k=1, telemetry_root=tmp_path / ".wmh")
    session = wm.new_session(task="secret task")
    wm.step(
        session.id,
        Action(kind=ActionKind.TOOL_CALL, name="secret_tool", arguments={"secret": "value"}),
    )

    serialized = json.dumps(calls)
    assert "wmh generated trace started" in serialized
    assert "wmh generated step completed" in serialized
    assert "generated_trace_count" in serialized
    assert "generated_step_count" in serialized
    assert "secret task" not in serialized
    assert "secret_tool" not in serialized
    assert "secret observation" not in serialized
    assert "value" not in serialized


def test_load_reads_artifact(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
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


def test_load_named_model_uses_project_root_for_telemetry_opt_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[object] = []

    class FakePosthog:
        def __init__(self, project_api_key: str, **kwargs: object) -> None:
            pass

        def capture(self, event: str, **kwargs: object) -> str:
            calls.append({"event": event, **kwargs})
            return "message-id"

        def shutdown(self) -> None:
            pass

    project_root = tmp_path / ".wmh"
    model_dir = project_root / "models" / "demo"
    save_config(HarnessConfig(top_k=2, embed_dim=64), model_dir)
    set_telemetry_enabled(False, project_root)
    telemetry._CLIENTS.clear()
    monkeypatch.setattr(telemetry, "Posthog", FakePosthog)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("WMH_POSTHOG_PROJECT_API_KEY", "phc_test")

    wm = WorldModel.load(str(model_dir), FakeProvider("{}"))
    wm.new_session(task="should respect project opt-out")

    assert wm._telemetry_root == project_root
    assert calls == []


def test_score_session_meters_judge_separately_from_serve() -> None:
    """Reward-judge tokens land under Phase.JUDGE on the session tracker, not SERVE (D12 split)."""
    from wmh.optimize.reward import EpisodeScore
    from wmh.tracking import Phase

    class JudgeReply:
        def __init__(self) -> None:
            self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="judge-m")

        def complete(
            self,
            system: str,
            messages: list[Message],
            *,
            temperature: float = 0.7,
            max_tokens: int = 8192,
        ) -> Completion:
            return Completion(
                text='{"success": false, "reward": 0.2, "step_rewards": [0.2], "critique": "c"}',
                usage=TokenUsage(input_tokens=10, output_tokens=5),
            )

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] for _ in texts]

        def verify(self) -> VerifyResult:
            raise NotImplementedError

    env_provider = FakeProvider('{"output": "found u1", "is_error": false}')
    retriever = _retriever_with(
        [
            Step(
                action=Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={}),
                observation=Observation(content="found u1"),
            )
        ]
    )
    wm = WorldModel(env_provider, retriever, top_k=1, reward_provider=JudgeReply())
    session = wm.new_session(task="do the thing")
    wm.step(session.id, Action(kind=ActionKind.TOOL_CALL, name="get_user", arguments={}))
    score = wm.score_session(session.id)
    assert isinstance(score, EpisodeScore)
    assert score.reward == 0.2
    assert session.history[-1].observation.reward == 0.2
    usage = wm.session_usage(session.id)
    assert usage.by_phase[Phase.JUDGE].input_tokens == 10  # judge cost split out (D12)
    assert Phase.SERVE in usage.by_phase  # the step call stays attributed to SERVE
