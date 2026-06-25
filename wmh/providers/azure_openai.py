"""Azure OpenAI provider (GPT 5.5).

Reads AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT from the environment; deployment name and
api_version come from ProviderConfig.deployment / ProviderConfig.api_version.
"""

from __future__ import annotations

from wmh.providers.base import Completion, Message, ProviderConfig, VerifyResult


class AzureOpenAIProvider:
    """GPT 5.5 via an Azure OpenAI deployment."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        # TODO: openai.AzureOpenAI(api_key, azure_endpoint, api_version) from env + config fields.

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        # TODO: chat.completions.create with system folded into messages; parse choice.
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        # TODO: embeddings.create against the configured embed deployment.
        raise NotImplementedError

    def verify(self) -> VerifyResult:
        raise NotImplementedError
