"""AWS Bedrock provider (Claude 4.8). Reads AWS_REGION + AWS credentials from the environment."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, TypedDict, cast

from wmh.providers.base import (
    Completion,
    Message,
    ProviderConfig,
    ProviderKind,
    TokenUsage,
    VerifyResult,
)

if TYPE_CHECKING:
    from botocore.client import BaseClient

# Bedrock speaks the same Anthropic Messages schema as the direct API, pinned by this version tag.
_ANTHROPIC_BEDROCK_VERSION = "bedrock-2023-05-31"


class _ContentBlock(TypedDict):
    type: str
    text: str


class _Usage(TypedDict):
    input_tokens: int
    output_tokens: int


class _BedrockResponse(TypedDict):
    content: list[_ContentBlock]
    usage: _Usage


class BedrockProvider:
    """Claude 4.8 via the Bedrock Runtime (InvokeModel with the Anthropic Messages body)."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client: BaseClient | None = None

    def _get_client(self) -> BaseClient:
        # Lazy: import boto3 and open the client only on first use. region falls back to
        # AWS_REGION / the default boto3 chain when config.region is unset.
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.config.region)
        return self._client

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        # Claude 4.8 rejects sampling params, so temperature is intentionally not forwarded.
        body = {
            "anthropic_version": _ANTHROPIC_BEDROCK_VERSION,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        raw = self._get_client().invoke_model(modelId=self.config.model, body=json.dumps(body))
        data = cast("_BedrockResponse", json.loads(raw["body"].read()))
        text = "".join(block["text"] for block in data["content"] if block["type"] == "text")
        usage = TokenUsage(
            input_tokens=data["usage"]["input_tokens"],
            output_tokens=data["usage"]["output_tokens"],
        )
        return Completion(text=text, usage=usage)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Embeddings on Bedrock (Titan / Cohere) are a separate model surface, not yet wired up.
        raise NotImplementedError(
            "BedrockProvider embeddings are not implemented; configure an OpenAI embed_provider."
        )

    def verify(self) -> VerifyResult:
        try:
            self.complete("", [Message(role="user", content="ping")], max_tokens=1)
        except Exception as exc:  # noqa: BLE001 - verify reports failure, never raises
            return VerifyResult(
                ok=False, kind=ProviderKind.BEDROCK, model=self.config.model, detail=str(exc)
            )
        return VerifyResult(ok=True, kind=ProviderKind.BEDROCK, model=self.config.model)
