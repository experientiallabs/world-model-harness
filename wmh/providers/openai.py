"""OpenAI direct provider (GPT 5.5). Reads OPENAI_API_KEY from the environment."""

from __future__ import annotations

from wmh.providers.base import Completion, Message, ProviderConfig, VerifyResult


class OpenAIProvider:
    """GPT 5.5 via the OpenAI API."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        # TODO: lazily construct openai.OpenAI() from OPENAI_API_KEY.

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def verify(self) -> VerifyResult:
        raise NotImplementedError
