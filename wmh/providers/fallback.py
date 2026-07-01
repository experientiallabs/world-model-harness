"""A Provider that fails over across a chain of backends on capacity errors.

Wraps an ordered list of providers and, per call, tries each in turn: the first is used until it
raises a *capacity* error (throttling / model-unavailable / timeout), at which point the next takes
over for that call. Non-capacity errors (bad request, auth) propagate immediately — failing over on
those would just mask a real bug.

Used to drive long GEPA runs off the preferred model while degrading gracefully to a fallback when
the preferred one is capacity-constrained, instead of aborting the whole (expensive) run.
"""

from __future__ import annotations

from wmh.providers.base import (
    Completion,
    Message,
    Provider,
    ProviderConfig,
    VerifyResult,
    verify_via_ping,
)

# Substrings that mark a *capacity* failure (retry on the next provider) rather than a client error
# (propagate). Bedrock surfaces these as botocore ClientError codes / messages.
_CAPACITY_MARKERS = (
    "throttl",
    "too many requests",
    "serviceunavailable",
    "service unavailable",
    "modelnotready",
    "model not ready",
    "capacity",
    "timeout",
    "timed out",
    "503",
    "429",
)


def _is_capacity_error(exc: Exception) -> bool:
    return any(marker in str(exc).lower() for marker in _CAPACITY_MARKERS)


class FallbackProvider:
    """Try `providers` in order per call; fail over only on capacity errors.

    `config` reports the *primary* provider's config (so cost/label attribution reads as the model
    we intend to use). Every wrapped provider must target the same logical role (all completion
    models); mixing embed models is out of scope — `embed` just delegates to the primary.
    """

    def __init__(self, providers: list[Provider]) -> None:
        if not providers:
            raise ValueError("FallbackProvider needs at least one provider")
        self._providers = providers
        self.config: ProviderConfig = providers[0].config

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        last_capacity_exc: Exception | None = None
        for provider in self._providers:
            try:
                return provider.complete(
                    system, messages, temperature=temperature, max_tokens=max_tokens
                )
            except Exception as exc:  # noqa: BLE001 - classify, then re-raise or fail over
                if not _is_capacity_error(exc):
                    raise  # a real error (bad request/auth) — don't mask it behind a fallback
                last_capacity_exc = exc
                continue  # capacity-constrained: try the next provider in the chain
        # Every provider was capacity-constrained.
        assert last_capacity_exc is not None
        raise last_capacity_exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._providers[0].embed(texts)

    def verify(self) -> VerifyResult:
        return verify_via_ping(self)
