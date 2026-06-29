"""Unified LLM provider layer.

One interface (`Provider`), four backends, one entry point (`get_provider`). All four can be
verified on startup with a cheap ping. Built fresh for this repo; no external client framework.
"""

from wmh.providers.base import (
    DEFAULT_MAX_TOKENS,
    Completion,
    EmbedderKind,
    Message,
    Provider,
    ProviderConfig,
    ProviderKind,
    VerifyResult,
)
from wmh.providers.registry import get_provider, verify_all, verify_embedder

__all__ = [
    "Provider",
    "ProviderConfig",
    "ProviderKind",
    "EmbedderKind",
    "DEFAULT_MAX_TOKENS",
    "Completion",
    "Message",
    "VerifyResult",
    "get_provider",
    "verify_all",
    "verify_embedder",
]
