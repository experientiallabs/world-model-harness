"""Tests for the offline HashingEmbedder."""

from __future__ import annotations

import math

import pytest

from wmh.retrieval.embedders import HashingEmbedder


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def test_embedding_is_deterministic_and_normalized() -> None:
    emb = HashingEmbedder(dim=128)
    v1 = emb.embed(["get_user u_kath"])[0]
    v2 = emb.embed(["get_user u_kath"])[0]
    assert v1 == v2
    assert len(v1) == 128
    assert _cosine(v1, v1) == pytest.approx(1.0)


def test_similar_text_is_closer_than_dissimilar() -> None:
    emb = HashingEmbedder(dim=512)
    base = emb.embed(["get_reservation r_042"])[0]
    similar = emb.embed(["get_reservation r_043"])[0]
    different = emb.embed(["issue_refund o_999 amount=500"])[0]
    assert _cosine(base, similar) > _cosine(base, different)


def test_batch_matches_individual() -> None:
    emb = HashingEmbedder(dim=64)
    batch = emb.embed(["a", "b"])
    assert batch[0] == emb.embed(["a"])[0]
    assert batch[1] == emb.embed(["b"])[0]


def test_rejects_nonpositive_dim() -> None:
    with pytest.raises(ValueError, match="positive"):
        HashingEmbedder(dim=0)
