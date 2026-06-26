"""Local, dependency-free embedders for retrieval (phi).

Bedrock's `embed` is intentionally not wired up (see `wmh.providers.bedrock`), and we don't require
an OpenAI key just to retrieve. `HashingEmbedder` provides a deterministic, offline phi so the whole
build/serve loop runs on completions alone: it is a classic hashed-bag-of-character-ngrams vector
(the "hashing trick"), L2-normalized. It captures lexical overlap between (state, action) renderings
— enough for top-k similarity over a trace replay buffer — without a model or network.

It implements the `embed` half of the `Provider` protocol; the world model uses a real completion
provider (e.g. Bedrock Opus) for generation and a `HashingEmbedder` for retrieval.
"""

from __future__ import annotations

import hashlib

import numpy as np

DEFAULT_DIM = 512
_NGRAM = 3


class HashingEmbedder:
    """Deterministic offline embedder: hashed character-trigram bag, L2-normalized.

    Not semantic, but stable and zero-dependency: identical text always maps to the identical
    vector, and lexically similar (state, action) pairs land near each other under cosine — which is
    what the retriever ranks on.
    """

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        if dim <= 0:
            raise ValueError(f"embedding dim must be positive, got {dim}")
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self._dim, dtype=np.float64)
        normalized = text.lower()
        if len(normalized) < _NGRAM:
            normalized = normalized.ljust(_NGRAM)
        for i in range(len(normalized) - _NGRAM + 1):
            gram = normalized[i : i + _NGRAM]
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self._dim
            # Sign from one extra bit so colliding grams can cancel rather than only accumulate.
            sign = 1.0 if digest[0] & 1 else -1.0
            vec[bucket] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()
