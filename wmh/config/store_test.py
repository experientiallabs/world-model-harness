"""Tests for the named-world-model store (resolution, listing, summaries)."""

from __future__ import annotations

import json

import pytest

from wmh.config import ArtifactPaths, HarnessConfig, save_config
from wmh.config.store import DEFAULT_MODEL_NAME, WorldModelStore, validate_name
from wmh.providers.base import ProviderConfig, ProviderKind


def _build_fake_model(store: WorldModelStore, name: str, accuracy: float = 0.5) -> None:
    """Write a minimal but valid artifact (config + metrics + frontier) for `name`."""
    root = store.model_dir(name)
    config = HarnessConfig(
        providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model="opus")],
        serve_provider=ProviderKind.BEDROCK,
    )
    save_config(config, root)
    paths = ArtifactPaths(root)
    paths.metrics.write_text(
        json.dumps({"held_out_accuracy": accuracy, "rollouts_used": 7}), encoding="utf-8"
    )
    paths.frontier.parent.mkdir(parents=True, exist_ok=True)
    paths.frontier.write_text(json.dumps(["a", "b"]), encoding="utf-8")


def test_validate_name_accepts_safe_names_and_rejects_traversal() -> None:
    assert validate_name("tau2-airline") == "tau2-airline"
    assert validate_name("retail.v2") == "retail.v2"
    for bad in ["../escape", "a/b", ".", "", ".hidden", "with space"]:
        with pytest.raises(ValueError, match="invalid world model name"):
            validate_name(bad)


def test_list_names_and_info(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    store = WorldModelStore(tmp_path / ".wmh", bundled_dir=None)
    assert store.list_names() == []
    _build_fake_model(store, "beta", accuracy=0.4)
    _build_fake_model(store, "alpha", accuracy=0.9)

    assert store.list_names() == ["alpha", "beta"]  # sorted
    info = store.info("alpha")
    assert info.serve_provider == "bedrock"
    assert info.serve_model == "opus"
    assert info.held_out_accuracy == 0.9
    assert info.rollouts_used == 7
    assert info.frontier_size == 2


def test_resolve_explicit_and_singleton_and_ambiguous(tmp_path) -> None:  # noqa: ANN001
    store = WorldModelStore(tmp_path / ".wmh", bundled_dir=None)

    # No models yet: resolve(None) errors helpfully.
    with pytest.raises(FileNotFoundError, match="no world models built"):
        store.resolve(None)

    _build_fake_model(store, DEFAULT_MODEL_NAME)
    # Exactly one model: resolve(None) picks it.
    assert store.resolve(None) == store.model_dir(DEFAULT_MODEL_NAME)
    # Explicit name resolves to its dir.
    assert store.resolve(DEFAULT_MODEL_NAME) == store.model_dir(DEFAULT_MODEL_NAME)

    _build_fake_model(store, "second")
    # Two models: resolve(None) is ambiguous.
    with pytest.raises(ValueError, match="multiple world models"):
        store.resolve(None)


def test_resolve_unknown_name_lists_available(tmp_path) -> None:  # noqa: ANN001
    store = WorldModelStore(tmp_path / ".wmh", bundled_dir=None)
    _build_fake_model(store, "alpha")
    with pytest.raises(FileNotFoundError, match="alpha"):
        store.resolve("nope")


def _build_bundled_model(bundled_dir, name: str, accuracy: float = 0.5) -> None:  # noqa: ANN001
    """Write a bundled artifact directly under `bundled_dir/<name>/` (no `models/` subdir)."""
    root = bundled_dir / name
    config = HarnessConfig(
        providers=[ProviderConfig(kind=ProviderKind.BEDROCK, model="opus")],
        serve_provider=ProviderKind.BEDROCK,
    )
    save_config(config, root)
    ArtifactPaths(root).metrics.write_text(
        json.dumps({"held_out_accuracy": accuracy, "rollouts_used": 7}), encoding="utf-8"
    )


def test_bundled_models_are_listed_and_resolved(tmp_path) -> None:  # noqa: ANN001
    bundled = tmp_path / "world-models"
    _build_bundled_model(bundled, "tau-bench")
    store = WorldModelStore(tmp_path / ".wmh", bundled_dir=bundled)

    # A bundled model is visible and resolvable even with an empty writable root.
    assert store.list_names() == ["tau-bench"]
    assert store.exists("tau-bench")
    assert store.resolve("tau-bench") == bundled / "tau-bench"
    # resolve(None) picks the lone bundled model.
    assert store.resolve(None) == bundled / "tau-bench"


def test_writable_model_shadows_bundled_of_same_name(tmp_path) -> None:  # noqa: ANN001
    bundled = tmp_path / "world-models"
    _build_bundled_model(bundled, "tau-bench", accuracy=0.1)
    store = WorldModelStore(tmp_path / ".wmh", bundled_dir=bundled)
    _build_fake_model(store, "tau-bench", accuracy=0.9)  # writes to .wmh/models/

    # Listed once (deduped), and the writable copy wins for both dir resolution and info.
    assert store.list_names() == ["tau-bench"]
    assert store.resolve("tau-bench") == store.model_dir("tau-bench")
    assert store.info("tau-bench").held_out_accuracy == 0.9


def test_writable_and_bundled_union_is_ambiguous(tmp_path) -> None:  # noqa: ANN001
    bundled = tmp_path / "world-models"
    _build_bundled_model(bundled, "tau-bench")
    store = WorldModelStore(tmp_path / ".wmh", bundled_dir=bundled)
    _build_fake_model(store, "my-own")

    assert store.list_names() == ["my-own", "tau-bench"]
    with pytest.raises(ValueError, match="multiple world models"):
        store.resolve(None)


def test_missing_bundled_dir_is_ignored(tmp_path) -> None:  # noqa: ANN001
    store = WorldModelStore(tmp_path / ".wmh", bundled_dir=tmp_path / "does-not-exist")
    assert store.bundled_dir is None
    assert store.list_names() == []
