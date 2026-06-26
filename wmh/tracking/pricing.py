"""Per-model token pricing → USD cost.

Provider-agnostic: prices are keyed by a normalized model id (provider prefixes like Bedrock's
`us.anthropic.` are stripped before lookup), so the same Opus 4.8 row covers the direct API and
Bedrock. Prices are USD per 1M tokens; an unknown model costs 0.0 and is flagged so callers can
surface "cost unavailable" rather than silently under-reporting.
"""

from __future__ import annotations

from pydantic import BaseModel

from wmh.providers.base import TokenUsage


class ModelPrice(BaseModel):
    """USD per 1,000,000 tokens, split by input/output."""

    input_per_mtok: float
    output_per_mtok: float


# Keyed by normalized model id (see `_normalize`). USD / 1M tokens.
# Sources: Claude API pricing (Opus 4.8 $5/$25, Sonnet 4.6 $3/$15, Haiku 4.5 $1/$5); OpenAI
# GPT-5.x and embedding list prices; Bedrock Titan v2 embeddings.
_PRICES: dict[str, ModelPrice] = {
    "claude-opus-4-8": ModelPrice(input_per_mtok=5.0, output_per_mtok=25.0),
    "claude-opus-4-7": ModelPrice(input_per_mtok=5.0, output_per_mtok=25.0),
    "claude-opus-4-6": ModelPrice(input_per_mtok=5.0, output_per_mtok=25.0),
    "claude-sonnet-4-6": ModelPrice(input_per_mtok=3.0, output_per_mtok=15.0),
    "claude-haiku-4-5": ModelPrice(input_per_mtok=1.0, output_per_mtok=5.0),
    "gpt-5.5": ModelPrice(input_per_mtok=5.0, output_per_mtok=25.0),
    "gpt-5.2": ModelPrice(input_per_mtok=5.0, output_per_mtok=25.0),
    "gpt-5.1": ModelPrice(input_per_mtok=5.0, output_per_mtok=25.0),
    "gpt-5": ModelPrice(input_per_mtok=5.0, output_per_mtok=25.0),
    # Embeddings (output tokens are always 0 for embed calls).
    "text-embedding-3-small": ModelPrice(input_per_mtok=0.02, output_per_mtok=0.0),
    "text-embedding-3-large": ModelPrice(input_per_mtok=0.13, output_per_mtok=0.0),
    "amazon.titan-embed-text-v2:0": ModelPrice(input_per_mtok=0.02, output_per_mtok=0.0),
}


def _normalize(model: str) -> str:
    """Strip provider/region routing prefixes so one row covers a model across providers.

    Bedrock ids look like `us.anthropic.claude-opus-4-8`; the direct API uses `claude-opus-4-8`.
    We drop a leading region segment (`us.`/`eu.`/...) and an `anthropic.` vendor segment, but keep
    `amazon.titan-...` (its `amazon.` is part of the canonical model id, not a routing prefix).
    """
    normalized = model.strip()
    region_prefixes = ("us.", "eu.", "apac.", "us-gov.")
    for prefix in region_prefixes:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    if normalized.startswith("anthropic."):
        normalized = normalized[len("anthropic.") :]
    return normalized


def price_for(model: str) -> ModelPrice | None:
    """Return the price row for `model` (after normalization), or None if unknown."""
    return _PRICES.get(_normalize(model))


def cost_usd(model: str, usage: TokenUsage) -> float:
    """USD cost of `usage` on `model`. Unknown models cost 0.0 (see `price_for` to detect that)."""
    price = price_for(model)
    if price is None:
        return 0.0
    return (
        usage.input_tokens * price.input_per_mtok + usage.output_tokens * price.output_per_mtok
    ) / 1_000_000
