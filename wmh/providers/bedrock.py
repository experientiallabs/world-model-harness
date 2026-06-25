"""AWS Bedrock provider (Claude 4.8). Reads AWS_REGION + AWS credentials from the environment."""

from __future__ import annotations

from wmh.providers.base import Completion, Message, ProviderConfig, VerifyResult


class BedrockProvider:
    """Claude 4.8 via the Bedrock Runtime (InvokeModel / Anthropic Messages on Bedrock)."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        # TODO: lazily construct boto3.client("bedrock-runtime", region_name=...).

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        # TODO: invoke_model with the anthropic messages body; parse response.
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        # TODO: Titan / Cohere embeddings on Bedrock, or delegate.
        raise NotImplementedError

    def verify(self) -> VerifyResult:
        raise NotImplementedError
