"""Tests for FallbackProvider: fail over on capacity errors, propagate real errors."""

from __future__ import annotations

import pytest

from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind
from wmh.providers.fallback import FallbackProvider, _is_capacity_error


class _StubProvider:
    """Returns `text`, or raises `raises` (an Exception) on complete()."""

    def __init__(self, name: str, *, text: str = "", raises: Exception | None = None) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model=name)
        self._text = text
        self._raises = raises
        self.calls = 0

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return Completion(text=self._text)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def _msg() -> list[Message]:
    return [Message(role="user", content="hi")]


def test_uses_primary_when_healthy() -> None:
    primary = _StubProvider("opus-4-6", text="from-primary")
    backup = _StubProvider("opus-4-7", text="from-backup")
    fb = FallbackProvider([primary, backup])
    assert fb.complete("s", _msg()).text == "from-primary"
    assert backup.calls == 0  # never touched


def test_fails_over_on_capacity_error() -> None:
    primary = _StubProvider("opus-4-6", raises=RuntimeError("ThrottlingException: slow down"))
    backup = _StubProvider("opus-4-7", text="from-backup")
    fb = FallbackProvider([primary, backup])
    assert fb.complete("s", _msg()).text == "from-backup"
    assert primary.calls == 1 and backup.calls == 1


def test_propagates_non_capacity_error() -> None:
    primary = _StubProvider("opus-4-6", raises=ValueError("malformed request: bad field"))
    backup = _StubProvider("opus-4-7", text="from-backup")
    fb = FallbackProvider([primary, backup])
    with pytest.raises(ValueError, match="malformed"):
        fb.complete("s", _msg())
    assert backup.calls == 0  # a real error must NOT silently fall through to the backup


def test_raises_last_capacity_error_when_all_constrained() -> None:
    p1 = _StubProvider("opus-4-6", raises=RuntimeError("throttled"))
    p2 = _StubProvider("opus-4-7", raises=RuntimeError("503 service unavailable"))
    fb = FallbackProvider([p1, p2])
    with pytest.raises(RuntimeError, match="service unavailable"):
        fb.complete("s", _msg())


def test_config_reports_primary() -> None:
    fb = FallbackProvider([_StubProvider("opus-4-6"), _StubProvider("opus-4-7")])
    assert fb.config.model == "opus-4-6"


def test_empty_chain_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FallbackProvider([])


def test_capacity_classifier() -> None:
    assert _is_capacity_error(RuntimeError("ThrottlingException"))
    assert _is_capacity_error(RuntimeError("429 Too Many Requests"))
    assert not _is_capacity_error(ValueError("ValidationException: bad model id"))
