"""Tests for the provider registry / entry point."""

from __future__ import annotations

import pytest

from wmh.providers import ProviderConfig, ProviderKind, get_provider
from wmh.providers.base import Provider


def test_all_four_providers_construct_and_satisfy_protocol() -> None:
    for kind in ProviderKind:
        provider = get_provider(ProviderConfig(kind=kind, model="m"))
        assert isinstance(provider, Provider)


def test_provider_verify_is_stubbed() -> None:
    provider = get_provider(ProviderConfig(kind=ProviderKind.ANTHROPIC, model="claude-opus-4-8"))
    with pytest.raises(NotImplementedError):
        provider.verify()
