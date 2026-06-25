"""OpenAI direct provider (GPT 5.5). Reads OPENAI_API_KEY from the environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wmh.providers import _openai_common
from wmh.providers.base import (
    Completion,
    Message,
    ProviderConfig,
    VerifyResult,
    verify_via_ping,
)

if TYPE_CHECKING:
    from openai import OpenAI


class OpenAIProvider:
    """GPT 5.5 via the OpenAI API."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        # Lazy: don't import the SDK or read OPENAI_API_KEY until first use.
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()  # picks up OPENAI_API_KEY from the environment
        return self._client

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        return _openai_common.complete(
            self._get_client().chat.completions, self.config.model, system, messages, max_tokens
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.config.embed_model is None:
            raise ValueError("OpenAIProvider.embed requires config.embed_model to be set.")
        return _openai_common.embed(self._get_client().embeddings, self.config.embed_model, texts)

    def verify(self) -> VerifyResult:
        return verify_via_ping(self)
