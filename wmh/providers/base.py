"""Provider interface and shared config/value types."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ProviderKind(StrEnum):
    ANTHROPIC = "anthropic"  # Opus 4.8 direct
    BEDROCK = "bedrock"  # Claude 4.8 via AWS
    AZURE_OPENAI = "azure_openai"  # GPT 5.5 via Azure
    OPENAI = "openai"  # GPT 5.5 direct


Role = Literal["user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class Completion(BaseModel):
    text: str
    usage: TokenUsage = Field(default_factory=TokenUsage)


class VerifyResult(BaseModel):
    ok: bool
    kind: ProviderKind
    model: str
    detail: str = ""


class ProviderConfig(BaseModel):
    """Everything needed to construct one provider.

    Credentials are read from the environment by default (keys named per backend); the explicit
    backend knobs below override. The env var names are documented in `wmh.config`.
    """

    kind: ProviderKind
    model: str
    embed_model: str | None = None
    # Backend knobs (only some apply per kind):
    endpoint: str | None = None  # Azure OpenAI / custom base URL
    region: str | None = None  # AWS Bedrock region
    deployment: str | None = None  # Azure OpenAI deployment name
    api_version: str | None = None  # Azure OpenAI API version


@runtime_checkable
class Provider(Protocol):
    """The single interface all four backends implement."""

    config: ProviderConfig

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        """Generate a completion. Used by the world model, GEPA, the judge, and the demo agent."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts for retrieval (phi in DreamGym). May delegate to a sibling embed model."""
        ...

    def verify(self) -> VerifyResult:
        """Cheap creds/model check run on startup (`wmh providers verify`)."""
        ...
