"""Azure OpenAI provider (GPT 5.5).

Reads AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT from the environment; deployment name and
api_version come from ProviderConfig.deployment / ProviderConfig.api_version.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wmh.providers import _openai_common
from wmh.providers.base import (
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    VerifyResult,
)

if TYPE_CHECKING:
    from openai import AzureOpenAI


class AzureOpenAIProvider:
    """GPT 5.5 via an Azure OpenAI deployment."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client: AzureOpenAI | None = None

    def _get_client(self) -> AzureOpenAI:
        # Lazy: construct on first use. api_key + endpoint default to AZURE_OPENAI_API_KEY /
        # AZURE_OPENAI_ENDPOINT from the environment; api_version must be supplied by config.
        if self._client is None:
            from openai import AzureOpenAI

            if self.config.api_version is None:
                raise ValueError("AzureOpenAIProvider requires config.api_version to be set.")
            self._client = AzureOpenAI(
                api_version=self.config.api_version,
                azure_endpoint=self.config.endpoint or _require_endpoint(),
            )
        return self._client

    def _deployment(self) -> str:
        # On Azure, the `model` arg to the API is the deployment name, not the base model id.
        if self.config.deployment is None:
            raise ValueError("AzureOpenAIProvider requires config.deployment to be set.")
        return self.config.deployment

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        return _openai_common.complete(
            self._get_client().chat.completions, self._deployment(), system, messages, max_tokens
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.config.embed_model is None:
            raise ValueError("AzureOpenAIProvider.embed requires config.embed_model (deployment).")
        return _openai_common.embed(self._get_client().embeddings, self.config.embed_model, texts)

    def verify(self) -> VerifyResult:
        try:
            self.complete("", [Message(role="user", content="ping")], max_tokens=1)
        except Exception as exc:  # noqa: BLE001 - verify reports failure, never raises
            return VerifyResult(
                ok=False, kind=ProviderKind.AZURE_OPENAI, model=self.config.model, detail=str(exc)
            )
        return VerifyResult(ok=True, kind=ProviderKind.AZURE_OPENAI, model=self.config.model)


def _require_endpoint() -> str:
    import os

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise ValueError(
            "AzureOpenAIProvider needs an endpoint: set config.endpoint or AZURE_OPENAI_ENDPOINT."
        )
    return endpoint
