"""Centralized provider entry point.

`get_provider` is the single constructor the rest of the harness uses; nothing imports a concrete
backend directly. `verify_all` powers `wmh providers verify`.
"""

from __future__ import annotations

from wmh.providers.anthropic import AnthropicProvider
from wmh.providers.azure_openai import AzureOpenAIProvider
from wmh.providers.base import Provider, ProviderConfig, ProviderKind, VerifyResult
from wmh.providers.bedrock import BedrockProvider
from wmh.providers.openai import OpenAIProvider

_BACKENDS = {
    ProviderKind.ANTHROPIC: AnthropicProvider,
    ProviderKind.BEDROCK: BedrockProvider,
    ProviderKind.AZURE_OPENAI: AzureOpenAIProvider,
    ProviderKind.OPENAI: OpenAIProvider,
}


def get_provider(config: ProviderConfig) -> Provider:
    """Construct the provider for `config.kind`. The one place backends are wired in."""
    try:
        backend = _BACKENDS[config.kind]
    except KeyError:  # pragma: no cover - exhaustive over the enum
        raise ValueError(f"unknown provider kind: {config.kind}") from None
    return backend(config)


def verify_all(configs: list[ProviderConfig]) -> list[VerifyResult]:
    """Ping every configured provider; never raises (failures come back as ok=False)."""
    results: list[VerifyResult] = []
    for cfg in configs:
        try:
            results.append(get_provider(cfg).verify())
        except Exception as exc:  # noqa: BLE001 - verification must not crash startup
            results.append(VerifyResult(ok=False, kind=cfg.kind, model=cfg.model, detail=str(exc)))
    return results
