"""Tests for config persistence and the artifact layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmh.config.config import HarnessConfig, load_config, save_config
from wmh.providers.base import ProviderConfig, ProviderKind


def test_save_then_load_round_trips(tmp_path: Path) -> None:
    config = HarnessConfig(
        providers=[
            ProviderConfig(kind=ProviderKind.ANTHROPIC, model="claude-opus-4-8"),
            ProviderConfig(
                kind=ProviderKind.AZURE_OPENAI,
                model="gpt-5.5",
                embed_model="text-embedding-3-large",
                endpoint="https://example.openai.azure.com",
                deployment="gpt-55",
                api_version="2024-02-01",
            ),
        ],
        serve_provider=ProviderKind.ANTHROPIC,
        embed_provider=ProviderKind.AZURE_OPENAI,
        top_k=8,
        train_split=0.7,
        gepa_budget=120,
        trace_adapter="otel-genai",
    )

    save_config(config, root=tmp_path / ".wmh")
    loaded = load_config(root=tmp_path / ".wmh")

    assert loaded == config


def test_save_creates_artifact_dir(tmp_path: Path) -> None:
    root = tmp_path / ".wmh"
    assert not root.exists()
    save_config(HarnessConfig(), root=root)
    assert (root / "config.toml").is_file()


def test_load_without_artifact_dir_raises_friendly_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="wmh build"):
        load_config(root=tmp_path / ".wmh")


def test_load_with_dir_but_no_config_raises_friendly_error(tmp_path: Path) -> None:
    root = tmp_path / ".wmh"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="config"):
        load_config(root=root)


def test_defaults_round_trip(tmp_path: Path) -> None:
    save_config(HarnessConfig(), root=tmp_path / ".wmh")
    assert load_config(root=tmp_path / ".wmh") == HarnessConfig()
