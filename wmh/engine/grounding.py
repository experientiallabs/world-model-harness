"""Web grounding: bounded search for entities the world model cannot ground internally.

When the env encounters a real-world entity outside its traces and knowledge base (an API's error
format, a package name, a flight code), it may emit a `ground_query` (see
`wmh.core.render.output_contract`) instead of hallucinating. A `Grounder` serves that query; the
engine caches results into the knowledge base (`grounded.md`) so an entity is searched at most
once per model, and re-completes the step with the results in context.

The default is `NullGrounder` — no network, tests and evals stay hermetic. Real backends:

- `BraveGrounder` (`BRAVE_SEARCH_API_KEY`; free tier at https://api-dashboard.search.brave.com/),
  a plain keyed JSON API with no scraping fragility — for free-text entity queries.
- `FetchGrounder` (keyless): when the agent's action is itself a read-only `curl` GET of a public
  URL, fetch that URL live and let the model shape the real body into the observation. Found
  empirically: 42% of the terminal-tasks test slice is curl-with-URL, scoring ~0.10 below every
  other step kind — the values (API payloads, search rankings) are unknowable without the network.
  Live fetches are inherently non-hermetic (the web has moved since capture); use only in serve or
  in explicitly-labeled eval modes.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, Field, ValidationError

from wmh.core.types import Action

GROUNDER_KINDS = ("none", "brave", "fetch")
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


class FetchGrounder:
    """Keyless grounder for URL-shaped queries: GET the URL, return its (capped) body.

    Non-URL queries yield no results — compose with a search backend if both are wanted. Fetches
    are memoized per instance (an eval or session asks about the same endpoint repeatedly) and
    failures return no results rather than raising: an unreachable URL should degrade to the
    ungrounded prediction, never break the step.
    """

    def __init__(self, *, max_chars: int = 8_000, fetch: FetchFn = _http_get) -> None:
        self._max_chars = max_chars
        self._fetch = fetch
        self._memo: dict[str, list[GroundingResult]] = {}

    def ground(self, query: str) -> list[GroundingResult]:
        url = query.strip()
        if not url.startswith(("http://", "https://")):
            return []
        if url in self._memo:
            return self._memo[url]
        try:
            body = self._fetch(url, {"Accept": "*/*", "User-Agent": "wmh-grounder"})
        except Exception:  # noqa: BLE001 - any transport failure degrades to "no grounding"
            results: list[GroundingResult] = []
        else:
            if len(body) > self._max_chars:
                body = body[: self._max_chars] + "\n[truncated]"
            results = [GroundingResult(title=url, url=url, snippet=body)]
        self._memo[url] = results
        return results


# curl flags that mean the request mutates state (or uploads); those commands are never fetched.
_CURL_MUTATING_FLAGS = re.compile(
    r"(^|\s)(-X\s*(?!GET\b)\w+|--request\s+(?!GET\b)\w+|-d\b|--data\b|--data-\w+|-F\b|--form\b"
    r"|-T\b|--upload-file\b)"
)
_URL_IN_COMMAND = re.compile(r"https?://[^\s\"'|;>]+")


def extract_get_url(action: Action) -> str | None:
    """Return the URL a read-only `curl` GET in `action` targets, or None.

    Conservative by design: only bash tool calls whose command invokes `curl` without any
    mutating/upload flag qualify — a fetch must be side-effect-free to be safe to execute for
    grounding. The first URL in the command is taken (pipes/filters after it are the model's job
    to apply to the fetched body).
    """
    if action.name != "bash":
        return None
    command = action.arguments.get("command")
    if not isinstance(command, str) or "curl" not in command:
        return None
    if _CURL_MUTATING_FLAGS.search(command):
        return None
    match = _URL_IN_COMMAND.search(command)
    return match.group(0) if match else None


def prefetched_knowledge(
    knowledge: str | None, action: Action, grounder: Grounder | None
) -> str | None:
    """Append a live fetch of `action`'s read-only GET URL to the knowledge text, when possible.

    The stateless prefetch used by replay experiments; the serving engine has its own budgeted,
    KB-cached variant (`WorldModel._predict`). Returns `knowledge` unchanged when there is no
    grounder, no fetchable URL, or the fetch yields nothing.
    """
    if grounder is None:
        return knowledge
    url = extract_get_url(action)
    if url is None:
        return knowledge
    results = grounder.ground(url)
    if not results:
        return knowledge
    block = f"## live fetch: {url}\n{render_grounding(results)}"
    return f"{knowledge}\n\n{block}" if knowledge else block


def get_grounder(kind: str) -> Grounder:
    """Construct the configured grounder (`HarnessConfig.grounder`): none | brave | fetch."""
    if kind == "none":
        return NullGrounder()
    if kind == "fetch":
        return FetchGrounder()
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
