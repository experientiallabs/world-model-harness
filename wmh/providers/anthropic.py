"""Anthropic direct provider (Opus 4.8). Reads ANTHROPIC_API_KEY from the environment."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from wmh.providers.base import (
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)

if TYPE_CHECKING:
    from anthropic import Anthropic
    from anthropic.types import MessageParam


class AnthropicProvider:
    """Primary backend: Opus 4.8 for env simulation, GEPA reflection, and the judge."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client: Anthropic | None = None

    def _get_client(self) -> Anthropic:
        # Lazy: don't import the SDK or read creds until first use, so the registry can
        # construct every backend without the optional `anthropic` extra installed.
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic()  # picks up ANTHROPIC_API_KEY from the environment
        return self._client

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        # Opus 4.8 takes `system` as a top-level arg and rejects sampling params, so temperature
        # is intentionally not forwarded; adaptive thinking is the default.
        api_messages = [
            cast("MessageParam", {"role": m.role, "content": m.content}) for m in messages
        ]
        response = self._get_client().messages.create(
            model=self.config.model,
            system=system,
            messages=api_messages,
            max_tokens=max_tokens,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return Completion(text=text, usage=usage)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Anthropic has no embeddings API.
        raise NotImplementedError(
            "AnthropicProvider has no embeddings API; set config.embed_provider to OpenAI or "
            "Bedrock for retrieval (phi)."
        )

    def verify(self) -> VerifyResult:
        try:
            self.complete("", [Message(role="user", content="ping")], max_tokens=1)
        except Exception as exc:  # noqa: BLE001 - verify reports failure, never raises
            return VerifyResult(
                ok=False, kind=ProviderKind.ANTHROPIC, model=self.config.model, detail=str(exc)
            )
        return VerifyResult(ok=True, kind=ProviderKind.ANTHROPIC, model=self.config.model)
