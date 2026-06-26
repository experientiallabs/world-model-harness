"""Tests for the token→cost pricing table."""

from __future__ import annotations

from wmh.providers.base import TokenUsage
from wmh.tracking.pricing import cost_usd, price_for


def test_opus_4_8_cost_is_5_in_25_out_per_mtok() -> None:
    # 1M input + 1M output on Opus 4.8 = $5 + $25.
    cost = cost_usd("claude-opus-4-8", TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000))
    assert cost == 30.0


def test_bedrock_prefix_normalizes_to_same_price() -> None:
    # The Bedrock-prefixed id prices identically to the direct id.
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=0)
    assert cost_usd("us.anthropic.claude-opus-4-8", usage) == cost_usd("claude-opus-4-8", usage)
    assert cost_usd("us.anthropic.claude-opus-4-8", usage) == 5.0


def test_titan_embedding_keeps_amazon_prefix() -> None:
    # `amazon.` is part of the Titan model id, not a routing prefix — it must still resolve.
    assert price_for("amazon.titan-embed-text-v2:0") is not None


def test_unknown_model_is_zero_and_flagged() -> None:
    assert price_for("totally-made-up-model") is None
    assert cost_usd("totally-made-up-model", TokenUsage(input_tokens=999, output_tokens=999)) == 0.0


def test_partial_usage_is_prorated() -> None:
    # 500k input + 200k output on Opus 4.8 = 0.5*5 + 0.2*25 = 2.5 + 5.0 = 7.5.
    cost = cost_usd("claude-opus-4-8", TokenUsage(input_tokens=500_000, output_tokens=200_000))
    assert cost == 7.5
