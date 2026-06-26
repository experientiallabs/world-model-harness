"""Reusable build/eval primitives for research ablations.

These wrap the real pipeline so an ablation measures the deployed behavior, not a reimplementation:

- `optimize_prompt` runs `GEPAOptimizer` at a chosen rollout `train_temperature` + GEPA `seed`
  (the knobs added to `wmh.optimize.gepa`) and returns the winning prompt + its metrics.
- `score_prompt` replay-scores a prompt's held-out reconstruction fidelity at a chosen
  `eval_temperature`, using the SAME leak-free RAG the serving model and `wmh eval` use.

Both take an explicit `Provider`, `Judge`, and `Embedder` so callers control whether they hit a live
backend (the `scripts/` runner) or fakes (the unit tests) — no network is assumed here.
"""

from __future__ import annotations

from wmh.core.types import Trace
from wmh.optimize.gepa import GEPAOptimizer, OptimizeResult, predict_observation
from wmh.optimize.judge import Judge
from wmh.providers.base import Embedder, Provider
from wmh.retrieval import EmbeddingRetriever
from wmh.retrieval.leakfree import DemoRetriever


def optimize_prompt(
    train: list[Trace],
    test: list[Trace],
    base_prompt: str,
    *,
    provider: Provider,
    judge: Judge,
    embedder: Embedder | None,
    budget: int,
    train_temperature: float,
    seed: int,
) -> OptimizeResult:
    """Evolve `base_prompt` with GEPA at `train_temperature` + `seed` (RAG-aware when `embedder`).

    Mirrors `wmh.engine.build`: a fresh train-only retriever makes optimization leak-free. Returns
    the GEPA `OptimizeResult` (winning prompt + held-out accuracy + rollouts used).
    """
    retriever = EmbeddingRetriever(embedder) if embedder is not None else None
    optimizer = GEPAOptimizer(
        provider,
        judge,
        retriever=retriever,
        temperature=train_temperature,
        seed=seed,
    )
    return optimizer.optimize(train, test, base_prompt, budget)


def score_prompt(
    prompt: str,
    held_out: list[Trace],
    *,
    provider: Provider,
    judge: Judge,
    embedder: Embedder | None,
    train: list[Trace] | None,
    eval_temperature: float,
    top_k: int = 5,
) -> float:
    """Replay-score `prompt`'s held-out fidelity at `eval_temperature` (leak-free RAG).

    Equivalent to `wmh.engine.replay.replay`, but with a configurable rollout temperature so an
    ablation can hold training fixed and vary only evaluation. Returns the mean judge score (0..1).
    A separate primitive (rather than calling `replay`) precisely because `replay` hardcodes the
    default temperature — surfacing the eval knob is the whole point of the experiment.
    """
    retriever = EmbeddingRetriever(embedder) if embedder is not None else None
    demos = DemoRetriever(retriever, train or [], top_k=top_k)
    scores: list[float] = []
    for trace in held_out:
        for step in trace.steps:
            predicted = predict_observation(
                provider,
                prompt,
                step.task,
                step.state_before,
                step.action,
                demos=demos.demos_for(trace.trace_id, step),
                temperature=eval_temperature,
            )
            scores.append(judge.score(predicted, step.observation, step).score)
    return sum(scores) / len(scores) if scores else 0.0
