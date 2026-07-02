"""Tests for the web-grounding seam."""

from __future__ import annotations

import json

import pytest

from wmh.engine.grounding import (
    BraveGrounder,
    GroundingResult,
    NullGrounder,
    get_grounder,
    render_grounding,
)


def test_null_grounder_never_searches() -> None:
    assert NullGrounder().ground("anything") == []


def test_get_grounder_maps_kinds() -> None:
    assert isinstance(get_grounder("none"), NullGrounder)
    with pytest.raises(ValueError, match="grounder"):
        get_grounder("bing")


def test_get_grounder_brave_without_key_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    with pytest.raises(ValueError, match="BRAVE_SEARCH_API_KEY"):
        get_grounder("brave")


def test_brave_grounder_parses_results_and_sends_key() -> None:
    seen: dict[str, str] = {}

    def fake_fetch(url: str, headers: dict[str, str]) -> str:
        seen["url"] = url
        seen.update(headers)
        return json.dumps(
            {
                "web": {
                    "results": [
                        {
                            "title": "tomli-w · PyPI",
                            "url": "https://pypi.org/project/tomli-w/",
                            "description": "A lil' TOML writer.",
                        }
                    ]
                }
            }
        )

    grounder = BraveGrounder(api_key="k123", fetch=fake_fetch)
    results = grounder.ground("tomli_w python package")
    assert results == [
        GroundingResult(
            title="tomli-w · PyPI",
            url="https://pypi.org/project/tomli-w/",
            snippet="A lil' TOML writer.",
        )
    ]
    assert "tomli_w+python+package" in seen["url"] or "tomli_w%20python%20package" in seen["url"]
    assert seen["X-Subscription-Token"] == "k123"


def test_brave_grounder_tolerates_missing_fields() -> None:
    def fake_fetch(url: str, headers: dict[str, str]) -> str:
        return json.dumps({"web": {"results": [{"title": "only title"}]}})

    results = BraveGrounder(api_key="k", fetch=fake_fetch).ground("q")
    assert results[0].title == "only title"
    assert results[0].url == "" and results[0].snippet == ""


def test_render_grounding_is_compact_markdown() -> None:
    text = render_grounding(
        [
            GroundingResult(title="t1", url="https://a", snippet="s1"),
            GroundingResult(title="t2", url="https://b", snippet="s2"),
        ]
    )
    assert "t1" in text and "https://b" in text and "s2" in text


def test_render_grounding_empty_says_so() -> None:
    assert "no results" in render_grounding([]).lower()
