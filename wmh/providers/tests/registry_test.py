"""Tests for the provider registry / entry point."""

from __future__ import annotations

import pytest

from wmh.providers import ProviderConfig, ProviderKind, get_provider
from wmh.providers.base import Provider


def test_all_four_providers_construct_and_satisfy_protocol() -> None:
    for kind in ProviderKind:
        provider = get_provider(ProviderConfig(kind=kind, model="m"))
        assert isinstance(provider, Provider)


def test_verify_never_raises_and_reports_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # No creds must surface as ok=False, never an exception — verify_all relies on this so
    # startup never crashes. Drop the key so this is deterministic regardless of the dev env.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    provider = get_provider(ProviderConfig(kind=ProviderKind.ANTHROPIC, model="claude-opus-4-8"))
    result = provider.verify()
    assert result.ok is False
    assert result.kind is ProviderKind.ANTHROPIC
