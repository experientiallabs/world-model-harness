"""Tests for the build pipeline (ingest -> split -> index -> GEPA -> persist), no network."""

from __future__ import annotations

import json

from wmh.config import ArtifactPaths, HarnessConfig
from wmh.core.types import Trace
from wmh.engine.build import build, split_traces
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval import HashingEmbedder


class FakeProvider:
    """Canned world-model JSON for rollouts; a fixed 'improved' prompt for GEPA reflection."""

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
        if "improve the system prompt" in system:
            return Completion(text="IMPROVED ENV PROMPT")
        if "grade a world model" in system:  # the judge
            return Completion(text='{"score": 0.5, "critique": "be more specific"}')
        return Completion(text='{"output": "ok", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def test_split_traces_is_deterministic_and_partitions() -> None:
    traces = [Trace(trace_id=f"t{i}") for i in range(50)]
    a_train, a_test = split_traces(traces, 0.8)
    b_train, b_test = split_traces(list(reversed(traces)), 0.8)
    # Same assignment regardless of order; every trace lands in exactly one side.
    assert {t.trace_id for t in a_train} == {t.trace_id for t in b_train}
    assert len(a_train) + len(a_test) == 50
    assert 0 < len(a_train) < 50  # roughly an 80/20 split, both sides non-empty


def test_build_writes_a_loadable_artifact(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    # A tiny OTel JSONL with one tool-call step.
    span_llm = {
        "traceId": "a" * 32,
        "spanId": "s1",
        "name": "chat",
        "startTimeUnixNano": 1,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {"key": "gen_ai.tool.name", "value": {"stringValue": "get_user"}},
            {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": '{"id": "u1"}'}},
            {"key": "gen_ai.prompt", "value": {"stringValue": "look up u1"}},
        ],
    }
    span_tool = {
        "traceId": "a" * 32,
        "spanId": "s2",
        "name": "execute_tool",
        "startTimeUnixNano": 2,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
            {"key": "gen_ai.tool.message", "value": {"stringValue": "found u1"}},
        ],
    }
    traces_file = tmp_path / "traces.jsonl"
    traces_file.write_text(
        json.dumps(span_llm) + "\n" + json.dumps(span_tool) + "\n", encoding="utf-8"
    )

    root = tmp_path / ".wmh"
    config = HarnessConfig(
        providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model="m")],
        serve_provider=ProviderKind.BEDROCK,
        embed_dim=64,
        gepa_budget=4,
        train_split=0.5,
    )
    result = build(
        config,
        file=str(traces_file),
        root=str(root),
        serve_provider=FakeProvider(),
        embedder=HashingEmbedder(dim=64),
    )

    paths = ArtifactPaths(root)
    assert paths.config.exists()
    assert paths.optimized_prompt.read_text(encoding="utf-8")  # a non-empty winning prompt
    assert json.loads(paths.frontier.read_text(encoding="utf-8"))  # frontier persisted
    assert result.prompt
    # The index round-trips: a freshly loaded WorldModel can retrieve the indexed step.
    from wmh.engine.world_model import WorldModel

    wm = WorldModel.load(str(root), FakeProvider())
    assert wm.sample_steps(5)
