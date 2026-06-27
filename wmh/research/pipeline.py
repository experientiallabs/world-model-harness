"""Reusable build/eval primitives for research ablations.

These wrap the real pipeline so an ablation measures the deployed behavior, not a reimplementation:

- `optimize_prompt` runs `GEPAOptimizer` at a chosen GEPA `seed` (the knob added to
  `wmh.optimize.gepa`) and returns the winning prompt + its metrics.
- `score_prompt` replay-scores a prompt's held-out reconstruction fidelity by delegating to the
  canonical `wmh.engine.replay.replay` — the SAME scorer `wmh eval` uses — so an experiment's metric
  is directly comparable to the rest of the harness (and any rubric/judge upgrade lands here for
  free).

Both take an explicit `Provider`, `Judge`, and `Embedder` so callers control whether they hit a live
backend (the `scripts/` runner) or fakes (the unit tests) — no network is assumed here.

Note on temperature: the rollout temperature is intentionally NOT a knob here. Every shipped
provider runs Opus 4.8 / GPT 5.5, which reject sampling params, so a temperature sweep would be
inert. It is parked as a future direction (docs/research_directions.md) pending a sampling-capable
provider.
"""

from __future__ import annotations

from wmh.core.types import Trace
from wmh.engine.replay import replay
from wmh.optimize.gepa import GEPAOptimizer, OptimizeResult
from wmh.optimize.judge import Judge
from wmh.providers.base import Embedder, Provider
from wmh.retrieval import EmbeddingRetriever


def optimize_prompt(
    train: list[Trace],
    test: list[Trace],
    base_prompt: str,
    *,
    provider: Provider,
    judge: Judge,
    embedder: Embedder | None,
    budget: int,
    seed: int,
) -> OptimizeResult:
    """Evolve `base_prompt` with GEPA at `seed` (RAG-aware when `embedder` is set).

    Mirrors `wmh.engine.build`: a fresh train-only retriever makes optimization leak-free. Returns
    the GEPA `OptimizeResult` (winning prompt + held-out accuracy + rollouts used).
    """
    retriever = EmbeddingRetriever(embedder) if embedder is not None else None
    optimizer = GEPAOptimizer(provider, judge, retriever=retriever, seed=seed)
    return optimizer.optimize(train, test, base_prompt, budget)


def score_prompt(
    prompt: str,
    held_out: list[Trace],
    *,
    provider: Provider,
    judge: Judge,
    embedder: Embedder | None,
    train: list[Trace] | None,
    top_k: int = 5,
) -> float:
    """Replay-score `prompt`'s held-out fidelity, leak-free. Returns the mean judge score (0..1).

    Thin adapter over `wmh.engine.replay.replay`: builds the serving retriever from `embedder` and
    forwards the leak-free `train` corpus, then returns the aggregate `mean_score`. Using `replay`
    (not a private loop) means the rubric/judge the rest of the harness uses scores ablations too.
    """
    retriever = EmbeddingRetriever(embedder) if embedder is not None else None
    report = replay(
        prompt,
        held_out,
        provider,
        judge,
        retriever=retriever,
        train=train if embedder is not None else None,
        top_k=top_k,
    )
    return report.mean_score
