"""Anthropic direct provider (Opus 4.8). Reads ANTHROPIC_API_KEY from the environment."""

from __future__ import annotations

from wmh.providers.base import Completion, Message, ProviderConfig, VerifyResult


class AnthropicProvider:
    """Primary backend: Opus 4.8 for env simulation, GEPA reflection, and the judge."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        # TODO: lazily construct anthropic.Anthropic() from ANTHROPIC_API_KEY.

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        # TODO: call messages.create; map content blocks -> Completion.text.
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Anthropic has no embeddings API; delegate to a configured embed provider (e.g. Voyage).
        raise NotImplementedError

    def verify(self) -> VerifyResult:
        # TODO: one cheap max_tokens=1 ping; return ok/detail.
        raise NotImplementedError
