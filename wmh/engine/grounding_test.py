"""Tests for the web-grounding seam."""

from __future__ import annotations

import json

import pytest

from wmh.core.types import Action, ActionKind
from wmh.engine.grounding import (
    BraveGrounder,
    FetchGrounder,
    GroundingResult,
    NullGrounder,
    extract_get_url,
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


def test_extract_get_url_finds_curl_get_targets() -> None:
    a = _bash('curl -s "https://api.github.com/repos/octocat/Hello-World" | jq .id')
    assert extract_get_url(a) == "https://api.github.com/repos/octocat/Hello-World"
    # plain flag soup still yields the url
    a2 = _bash("curl -sL -H 'Accept: application/json' https://pypi.org/pypi/flask/json")
    assert extract_get_url(a2) == "https://pypi.org/pypi/flask/json"


def test_extract_get_url_rejects_mutating_or_non_curl_commands() -> None:
    assert extract_get_url(_bash('curl -X POST -d "x=1" https://api.example.com/things')) is None
    assert extract_get_url(_bash("curl --data foo https://api.example.com/things")) is None
    assert extract_get_url(_bash("curl -T file.txt https://api.example.com/up")) is None
    assert extract_get_url(_bash("wget https://example.com/file")) is None  # curl only, for now
    assert extract_get_url(_bash("echo hello")) is None
    assert extract_get_url(Action(kind=ActionKind.MESSAGE, content="curl https://x.dev")) is None


def test_fetch_grounder_gets_url_and_memoizes() -> None:
    calls: list[str] = []

    def fake_fetch(url: str, headers: dict[str, str]) -> str:
        calls.append(url)
        return '{"info": {"home_page": null}}'

    grounder = FetchGrounder(fetch=fake_fetch)
    results = grounder.ground("https://pypi.org/pypi/flask/json")
    assert results[0].url == "https://pypi.org/pypi/flask/json"
    assert '"home_page": null' in results[0].snippet
    grounder.ground("https://pypi.org/pypi/flask/json")  # second ask
    assert calls == ["https://pypi.org/pypi/flask/json"]  # memoized: one real fetch


def test_fetch_grounder_caps_body_and_swallows_fetch_errors() -> None:
    def big(url: str, headers: dict[str, str]) -> str:
        return "x" * 100_000

    capped = FetchGrounder(fetch=big, max_chars=500).ground("https://a.dev")
    assert len(capped[0].snippet) <= 520  # body cap + truncation marker

    def boom(url: str, headers: dict[str, str]) -> str:
        raise OSError("connection refused")

    assert FetchGrounder(fetch=boom).ground("https://b.dev") == []  # fail-safe: no results


def test_fetch_grounder_ignores_non_url_queries() -> None:
    def fail(url: str, headers: dict[str, str]) -> str:
        raise AssertionError("must not fetch")

    assert FetchGrounder(fetch=fail).ground("tomli_w python package") == []


def _bash(command: str) -> Action:
    return Action(kind=ActionKind.TOOL_CALL, name="bash", arguments={"command": command})
