"""Web grounding: bounded search for entities the world model cannot ground internally.

When the env encounters a real-world entity outside its traces and knowledge base (an API's error
format, a package name, a flight code), it may emit a `ground_query` (see
`wmh.core.render.output_contract`) instead of hallucinating. A `Grounder` serves that query; the
engine caches results into the knowledge base (`grounded.md`) so an entity is searched at most
once per model, and re-completes the step with the results in context.

The default is `NullGrounder` — no network, tests and evals stay hermetic. The one real backend is
Brave Search (`BRAVE_SEARCH_API_KEY`; free tier at https://api-dashboard.search.brave.com/), a
plain keyed JSON API with no scraping fragility.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

GROUNDER_KINDS = ("none", "brave")
_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_TIMEOUT_SECONDS = 15.0


class GroundingResult(BaseModel):
    """One search hit: enough to ground an entity, small enough to cache in the KB."""

    title: str = ""
    url: str = ""
    snippet: str = ""


class Grounder(Protocol):
    """Anything that can answer a grounding query with search results."""

    def ground(self, query: str) -> list[GroundingResult]:
        """Return search results for `query` (empty when grounding is unavailable)."""
        ...


class NullGrounder:
    """The default: grounding disabled, never touches the network."""

    def ground(self, query: str) -> list[GroundingResult]:
        return []


# Injectable HTTP GET (url, headers) -> response body; lets tests exercise BraveGrounder offline.
FetchFn = Callable[[str, dict[str, str]], str]


def _http_get(url: str, headers: dict[str, str]) -> str:
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 — https endpoint constant
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310
        body: str = response.read().decode("utf-8")
        return body


class _BraveWeb(BaseModel):
    results: list[GroundingResult] = Field(default_factory=list)


class _BraveResponse(BaseModel):
    web: _BraveWeb = Field(default_factory=_BraveWeb)


class BraveGrounder:
    """Brave Search API backend (`X-Subscription-Token` keyed GET, JSON response)."""

    def __init__(self, api_key: str, *, count: int = 5, fetch: FetchFn = _http_get) -> None:
        self._api_key = api_key
        self._count = count
        self._fetch = fetch

    def ground(self, query: str) -> list[GroundingResult]:
        params = urllib.parse.urlencode({"q": query, "count": str(self._count)})
        headers = {"Accept": "application/json", "X-Subscription-Token": self._api_key}
        body = self._fetch(f"{_BRAVE_ENDPOINT}?{params}", headers)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Brave search returned non-JSON for {query!r}: {body[:200]}") from exc
        try:
            # Brave's result objects use `description` for the snippet; map it before validation.
            raw_results = payload.get("web", {}).get("results", [])
            for item in raw_results:
                if isinstance(item, dict) and "description" in item and "snippet" not in item:
                    item["snippet"] = item.pop("description")
            return _BraveResponse.model_validate(payload).web.results
        except (ValidationError, AttributeError) as exc:
            raise ValueError(
                f"Brave search response for {query!r} did not match the expected shape: {exc}"
            ) from exc


def get_grounder(kind: str) -> Grounder:
    """Construct the configured grounder (`HarnessConfig.grounder`): "none" or "brave"."""
    if kind == "none":
        return NullGrounder()
    if kind == "brave":
        api_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
        if not api_key:
            raise ValueError(
                "grounder 'brave' needs BRAVE_SEARCH_API_KEY set; get a free key at "
                "https://api-dashboard.search.brave.com/ or set grounder = 'none'"
            )
        return BraveGrounder(api_key)
    raise ValueError(f"unknown grounder {kind!r}; choose one of {', '.join(GROUNDER_KINDS)}")


def render_grounding(results: list[GroundingResult]) -> str:
    """Render results as compact markdown for the KB cache and the re-completion prompt."""
    if not results:
        return "(no results)"
    return "\n".join(f"- {r.title} ({r.url}): {r.snippet}".strip() for r in results)
