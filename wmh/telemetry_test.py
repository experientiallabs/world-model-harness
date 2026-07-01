"""Tests for anonymous PostHog telemetry capture."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import httpx
import pytest

from wmh.config.settings import set_telemetry_enabled
from wmh.telemetry import capture


def test_capture_posts_anonymous_metadata_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, object, float]] = []

    def fake_post(url: str, *, json: object, timeout: float) -> httpx.Response:
        calls.append((url, json, timeout))
        return httpx.Response(200)

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv("WMH_TELEMETRY", "1")
    monkeypatch.setenv("WMH_POSTHOG_PROJECT_API_KEY", "phc_test")

    assert capture("wmh test event", {"generated_step_count": 1}, root=tmp_path / ".wmh")

    assert len(calls) == 1
    url, raw_payload, timeout = calls[0]
    payload = cast(dict[str, object], raw_payload)
    properties = cast(dict[str, object], payload["properties"])
    assert url == "https://us.i.posthog.com/i/v0/e/"
    assert timeout == 0.5
    assert payload["api_key"] == "phc_test"
    assert payload["event"] == "wmh test event"
    assert isinstance(payload["distinct_id"], str)
    assert properties["$process_person_profile"] is False
    assert properties["generated_step_count"] == 1


def test_capture_respects_project_opt_out(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[object] = []

    def fake_post(url: str, *, json: object, timeout: float) -> httpx.Response:
        calls.append((url, json, timeout))
        return httpx.Response(200)

    root = tmp_path / ".wmh"
    set_telemetry_enabled(False, root)
    monkeypatch.delenv("WMH_TELEMETRY", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("WMH_POSTHOG_PROJECT_API_KEY", "phc_test")
    monkeypatch.setattr(httpx, "post", fake_post)

    assert capture("wmh test event", root=root) is False
    assert calls == []


def test_do_not_track_wins_over_env_enable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[object] = []

    def fake_post(url: str, *, json: object, timeout: float) -> httpx.Response:
        calls.append((url, json, timeout))
        return httpx.Response(200)

    monkeypatch.setenv("WMH_TELEMETRY", "1")
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    monkeypatch.setattr(httpx, "post", fake_post)

    assert capture("wmh test event", root=tmp_path / ".wmh") is False
    assert calls == []
