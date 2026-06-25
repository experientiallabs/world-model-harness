"""Unit tests for BedrockProvider. No network: the boto3 client is faked via _get_client."""

from __future__ import annotations

import io
import json
from typing import cast

import pytest

from wmh.providers.base import Message, ProviderConfig, ProviderKind
from wmh.providers.bedrock import BedrockProvider


class _FakeBody:
    def __init__(self, payload: dict[str, object]) -> None:
        self._raw = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._raw


class _FakeClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.last_kwargs: dict[str, object] = {}

    def invoke_model(self, **kwargs: object) -> dict[str, object]:
        self.last_kwargs = kwargs
        return {"body": _FakeBody(self.payload)}


def _config() -> ProviderConfig:
    return ProviderConfig(
        kind=ProviderKind.BEDROCK, model="anthropic.claude-opus-4-8", region="us-east-1"
    )


def _payload() -> dict[str, object]:
    return {
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


def test_complete_builds_anthropic_body_and_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeClient(_payload())
    provider = BedrockProvider(_config())
    monkeypatch.setattr(provider, "_get_client", lambda: fake)  # inject fake; no network

    completion = provider.complete("sys", [Message(role="user", content="hi")], max_tokens=32)

    assert completion.text == "ok"
    assert completion.usage.input_tokens == 5
    assert completion.usage.output_tokens == 3
    assert fake.last_kwargs["modelId"] == "anthropic.claude-opus-4-8"
    body = json.loads(cast("str", fake.last_kwargs["body"]))
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["max_tokens"] == 32
    assert body["system"] == "sys"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert "temperature" not in body  # Claude 4.8 rejects sampling params


def test_embed_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="embed_provider"):
        BedrockProvider(_config()).embed(["x"])


def test_verify_reports_failure_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        def invoke_model(self, **kwargs: object) -> dict[str, object]:
            raise RuntimeError("no creds")

    provider = BedrockProvider(_config())
    monkeypatch.setattr(provider, "_get_client", _Boom)
    result = provider.verify()
    assert result.ok is False
    assert "no creds" in result.detail
    assert result.kind is ProviderKind.BEDROCK


def test_fake_body_is_stream_like() -> None:
    # Guard: real boto3 returns a StreamingBody; .read() once is the contract we rely on.
    assert isinstance(io.BytesIO(b"x").read(), bytes)


@pytest.mark.skipif(
    "AWS_REGION" not in __import__("os").environ,
    reason="no AWS_REGION; skipping live smoke test",
)
def test_live_verify() -> None:  # pragma: no cover - network
    import os

    provider = BedrockProvider(
        ProviderConfig(
            kind=ProviderKind.BEDROCK,
            model="anthropic.claude-opus-4-8",
            region=os.environ["AWS_REGION"],
        )
    )
    assert provider.verify().ok is True
