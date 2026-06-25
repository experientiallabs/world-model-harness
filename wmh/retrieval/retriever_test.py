"""Tests for the EmbeddingRetriever (DreamGym top-k replay buffer).

A deterministic fake Provider returns canned embeddings keyed off the encoded (state, action) text,
so we can assert exact top-k ordering by cosine similarity without any network.
"""

from __future__ import annotations

from wmh.core.types import Action, ActionKind, EnvState, Observation, Step, Trace
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.retrieval.retriever import EmbeddingRetriever, Retriever


class FakeEmbedProvider:
    """Maps known encoded texts to fixed vectors; unknown texts get an orthogonal default."""

    def __init__(self, table: dict[str, list[float]], default: list[float]) -> None:
        self.config = ProviderConfig(kind=ProviderKind.ANTHROPIC, model="m")
        self._table = table
        self._default = default
        self.embed_calls = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        return Completion(text="")

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += 1
        out: list[list[float]] = []
        for t in texts:
            vec = next((v for key, v in self._table.items() if key in t), self._default)
            out.append(list(vec))
        return out

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def _step(tool: str, arg: int, obs: str) -> Step:
    return Step(
        action=Action(kind=ActionKind.TOOL_CALL, name=tool, arguments={"x": arg}),
        observation=Observation(content=obs),
        state_before=EnvState(structured={"loc": tool}),
    )


def test_embedding_retriever_satisfies_protocol() -> None:
    provider = FakeEmbedProvider({}, default=[1.0, 0.0])
    assert isinstance(EmbeddingRetriever(provider), Retriever)


def test_topk_orders_by_cosine_similarity() -> None:
    # Three steps with distinct embeddings; query aligns most with "alpha", then "beta".
    table = {
        "alpha": [1.0, 0.0, 0.0],
        "beta": [0.8, 0.6, 0.0],
        "gamma": [0.0, 0.0, 1.0],
    }
    provider = FakeEmbedProvider(table, default=[1.0, 0.0, 0.0])  # query encodes to "alpha" tool
    retriever = EmbeddingRetriever(provider)
    steps = [_step("gamma", 1, "g"), _step("alpha", 2, "a"), _step("beta", 3, "b")]
    retriever.index([Trace(trace_id="t", steps=steps)])

    top = retriever.topk(EnvState(structured={"loc": "alpha"}), steps[1].action, k=2)
    assert [s.action.name for s in top] == ["alpha", "beta"]


def test_topk_respects_k_and_returns_at_most_buffer_size() -> None:
    provider = FakeEmbedProvider({"alpha": [1.0, 0.0]}, default=[1.0, 0.0])
    retriever = EmbeddingRetriever(provider)
    retriever.index([Trace(trace_id="t", steps=[_step("alpha", 1, "a")])])

    assert retriever.topk(EnvState(), _step("alpha", 1, "a").action, k=5) != []
    assert len(retriever.topk(EnvState(), _step("alpha", 1, "a").action, k=5)) == 1
    assert retriever.topk(EnvState(), _step("alpha", 1, "a").action, k=0) == []


def test_topk_on_empty_buffer_returns_empty() -> None:
    provider = FakeEmbedProvider({}, default=[1.0, 0.0])
    retriever = EmbeddingRetriever(provider)
    retriever.index([])
    assert retriever.topk(EnvState(), _step("alpha", 1, "a").action, k=3) == []


def test_add_enriches_buffer_online() -> None:
    table = {"alpha": [1.0, 0.0], "beta": [0.0, 1.0]}
    provider = FakeEmbedProvider(table, default=[1.0, 0.0])
    retriever = EmbeddingRetriever(provider)
    retriever.index([Trace(trace_id="t", steps=[_step("beta", 1, "b")])])

    added = _step("alpha", 9, "freshly added")
    retriever.add(added)
    top = retriever.topk(EnvState(structured={"loc": "alpha"}), added.action, k=1)
    assert top[0].observation.content == "freshly added"


def test_add_into_empty_buffer_then_retrieves() -> None:
    provider = FakeEmbedProvider({"alpha": [1.0, 0.0]}, default=[1.0, 0.0])
    retriever = EmbeddingRetriever(provider)
    retriever.index([])  # buffer starts empty
    step = _step("alpha", 1, "first")
    retriever.add(step)
    assert retriever.topk(EnvState(), step.action, k=1)[0].observation.content == "first"


def test_encode_text_is_structured_and_stable() -> None:
    state = EnvState(structured={"b": 2, "a": 1}, scratchpad="logged in")
    action = Action(kind=ActionKind.TOOL_CALL, name="buy", arguments={"sku": "A1"})
    text = EmbeddingRetriever._encode_text(state, action)
    # Keys are sorted -> deterministic rendering regardless of dict insertion order.
    assert "STATE:" in text and "ACTION kind=tool_call" in text
    assert "tool: buy" in text and '"a":1' in text and '"b":2' in text
    assert "scratchpad: logged in" in text
    other = EnvState(structured={"a": 1, "b": 2}, scratchpad="logged in")
    assert EmbeddingRetriever._encode_text(other, action) == text
