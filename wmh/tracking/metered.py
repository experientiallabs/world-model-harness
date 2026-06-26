"""`MeteredProvider`: a Provider wrapper that records every call onto a RunTracker.

Instrumenting at the provider boundary means the optimizer, the judge, and the world model are all
metered without any of them knowing about tracking — we don't edit `gepa.py` or the judge. The
wrapper forwards `complete`/`embed`/`verify`/`config` to the wrapped provider unchanged and records
a `UsageEvent` per call.

Phase attribution: build and serve share one provider, and within build the *same* provider serves
GEPA rollouts, GEPA reflection, and the judge. We tell them apart by the system prompt each path
uses (a stable, boundary-visible signal), defaulting `complete` to whatever base phase the wrapper
was constructed with. Callers that want exact control can pass their own `classify`.
"""

from __future__ import annotations

from collections.abc import Callable

from wmh.providers.base import (
    Completion,
    Message,
    Provider,
    ProviderConfig,
    TokenUsage,
    VerifyResult,
)
from wmh.tracking.tracker import Phase, RunTracker

# System-prompt fingerprints of the build-time call sites (see judge.py / gepa.py).
_JUDGE_MARKER = "grade a world model"
_REFLECTION_MARKER = "improve the system prompt"


def classify_build_call(system: str) -> Phase:
    """Default phase classifier for a build-time `complete`: judge vs GEPA (rollout/reflection)."""
    if _JUDGE_MARKER in system:
        return Phase.JUDGE
    # GEPA rollouts (env-sim) and reflection both belong to the optimization phase.
    return Phase.GEPA


class MeteredProvider:
    """Wraps a `Provider`, recording token usage + cost per call onto a `RunTracker`.

    `base_phase` is the phase for `complete` calls when no `classify` is given (e.g. `Phase.SERVE`
    for the live world model). For build, pass `classify=classify_build_call` to split judge from
    GEPA.
    """

    def __init__(
        self,
        provider: Provider,
        tracker: RunTracker,
        *,
        base_phase: Phase = Phase.OTHER,
        classify: Callable[[str], Phase] | None = None,
    ) -> None:
        self._provider = provider
        self._tracker = tracker
        self._base_phase = base_phase
        self._classify = classify

    @property
    def config(self) -> ProviderConfig:
        return self._provider.config

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        completion = self._provider.complete(
            system, messages, temperature=temperature, max_tokens=max_tokens
        )
        phase = self._classify(system) if self._classify is not None else self._base_phase
        self._tracker.record(phase, self._provider.config.model, completion.usage)
        return completion

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Embeddings carry no token usage from our providers; record a zero-usage event for the
        # call count so EMBED still shows up in the breakdown.
        vectors = self._provider.embed(texts)
        self._tracker.record(Phase.EMBED, self._provider.config.model, TokenUsage())
        return vectors

    def verify(self) -> VerifyResult:
        return self._provider.verify()
