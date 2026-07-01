#!/usr/bin/env python
"""Run GEPA with a proper train/val/TEST split and measure lift on the held-out TEST set.

The goal: confirm GEPA's evolved prompt beats the base prompt on traces it NEVER saw during
optimization (not the val set it selects candidates on). This is the honest generalization test the
old 2-way split couldn't give.

    AWS_REGION=us-west-1 uv run python scripts/gepa_test_lift.py examples/tau2-bench.otel.jsonl \
        --train 0.5 --val 0.25 --iterations 8 --out /tmp/gepa_lift.json

Uses the model fallback chain (Opus 4.6 -> 4.7 -> Sonnet 4.6 -> Opus 4.8) for every call, so a
capacity-constrained preferred model degrades gracefully instead of aborting the run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The requested fallback chain: prefer 4.6, then 4.7, then Sonnet 4.6, then 4.8 only as last resort.
FALLBACK_MODELS = [
    "us.anthropic.claude-opus-4-6-v1",
    "us.anthropic.claude-opus-4-7",
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-opus-4-8",
]


def build_provider(region: str):  # noqa: ANN201 - returns a Provider
    from wmh.providers import ProviderConfig, ProviderKind, get_provider
    from wmh.providers.fallback import FallbackProvider

    chain = [
        get_provider(ProviderConfig(kind=ProviderKind.BEDROCK, model=m, region=region))
        for m in FALLBACK_MODELS
    ]
    return FallbackProvider(chain)


def score_prompt(prompt, test_traces, train_traces, provider, embedder, max_tokens):  # noqa: ANN001, ANN201
    """Mean rubric fidelity of `prompt` on `test_traces`, with leak-free RAG from `train_traces`."""
    from wmh.engine.replay import replay
    from wmh.optimize.judge import RubricJudge
    from wmh.retrieval import EmbeddingRetriever

    retriever = EmbeddingRetriever(embedder)
    retriever.index(train_traces)
    report = replay(
        prompt,
        test_traces,
        provider,
        RubricJudge(provider),
        retriever=retriever,
        train=train_traces,
        top_k=5,
        sample_turns="all",
        seed=0,
        max_tokens=max_tokens,
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--train", type=float, default=0.5)
    ap.add_argument("--val", type=float, default=0.25)
    ap.add_argument("--iterations", type=int, default=8)
    ap.add_argument("--region", default="us-west-1")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument(
        "--hard-only",
        action="store_true",
        help="Restrict GEPA's reflection trainset to steps with prompt-addressable headroom "
        "(searches/lists + error observations), skipping easy cold lookups.",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from wmh.engine.build import split_traces_3way
    from wmh.engine.prompts import BASE_ENV_PROMPT
    from wmh.ingest import get_adapter
    from wmh.optimize import GEPAOptimizer, RubricJudge
    from wmh.retrieval import EmbeddingRetriever, HashingEmbedder

    traces = get_adapter("otel-genai").from_file(args.file)
    train, val, test = split_traces_3way(traces, args.train, args.val)
    print(
        f"corpus={len(traces)} traces -> train={len(train)} val={len(val)} test={len(test)} | "
        f"test steps={sum(len(t.steps) for t in test)}"
    )

    provider = build_provider(args.region)
    embedder = HashingEmbedder(dim=512)

    # GEPA optimizes on train (minibatches) + val (candidate selection). TEST is untouched.
    optimizer = GEPAOptimizer(
        provider, RubricJudge(provider), retriever=EmbeddingRetriever(embedder)
    )
    # Optional: focus reflection on steps with prompt-addressable headroom. Searches/lists can
    # over-populate empty results (fixable); error observations test success/error prediction
    # (fixable). Pure record lookups from empty state are data-bound (the model can't know the
    # values), so reflecting on them wastes iterations. The valset stays unfiltered.
    def _is_hard(step) -> bool:  # noqa: ANN001
        name = (step.action.name or "").lower()
        if any(k in name for k in ("search", "list", "find")):
            return True
        return step.observation.is_error

    hard_filter = _is_hard if args.hard_only else None
    if args.hard_only:
        n_hard = sum(_is_hard(s) for t in train for s in t.steps)
        print(f"hard-only: {n_hard} of {sum(len(t.steps) for t in train)} train steps kept")

    print(f"running GEPA: {args.iterations} iterations on train+val (test held out)...")
    result = optimizer.optimize(
        train, val, BASE_ENV_PROMPT, args.iterations, hard_step_filter=hard_filter
    )
    evolved = result.prompt
    changed = evolved.strip() != BASE_ENV_PROMPT.strip()
    print(f"GEPA done: prompt changed from base? {changed} | frontier={len(result.frontier)}")

    # Score BASE and EVOLVED on the held-out TEST set (same RAG corpus = train, leak-free).
    print("scoring BASE on test...")
    base_rep = score_prompt(BASE_ENV_PROMPT, test, train, provider, embedder, args.max_tokens)
    print(f"  base test fidelity = {base_rep.mean_score:.3f} +/- {base_rep.score_std:.3f}")
    print("scoring EVOLVED on test...")
    eff_rep = score_prompt(evolved, test, train, provider, embedder, args.max_tokens)
    print(f"  evolved test fidelity = {eff_rep.mean_score:.3f} +/- {eff_rep.score_std:.3f}")

    lift = eff_rep.mean_score - base_rep.mean_score
    print(
        f"\n=== TEST-SET LIFT: {lift:+.3f} "
        f"(base {base_rep.mean_score:.3f} -> evolved {eff_rep.mean_score:.3f}) ==="
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "prompt_changed": changed,
                "base": base_rep.model_dump(),
                "evolved": eff_rep.model_dump(),
                "lift": lift,
                "evolved_prompt": evolved,
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
