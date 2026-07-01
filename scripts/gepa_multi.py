#!/usr/bin/env python
"""Fast multi-candidate GEPA: pin baselines, spawn N GEPA candidates in parallel, rank on test.

Answers "does GEPA beat plain RAG?" per corpus:
  1. Pin two hard baselines on the held-out TEST set: base prompt (no RAG) and base+RAG.
  2. Spawn N GEPA candidates concurrently, each optimizing on a different train SUBSET (seeded), all
     with knowledge-accumulation reflection + merge, selecting on hard-step val fidelity.
  3. Score every evolved prompt on the SAME test set (with RAG) and report which beat base+RAG.

GEPA calls are Bedrock-latency-bound, so candidates run in a thread pool — wall-clock ≈ one GEPA run
rather than N. Every call goes through the fallback chain (Opus 4.6 -> 4.7 -> Sonnet 4.6 -> 4.8).

    AWS_REGION=us-west-1 uv run python scripts/gepa_multi.py CORPUS.otel.jsonl \
        --candidates 4 --iterations 6 --train 0.5 --val 0.25 --test-cap 40 --out /tmp/multi.json
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FALLBACK_MODELS = [
    "us.anthropic.claude-opus-4-6-v1",
    "us.anthropic.claude-opus-4-7",
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-opus-4-8",
]


def build_provider(region: str):  # noqa: ANN201
    from wmh.providers import ProviderConfig, ProviderKind, get_provider
    from wmh.providers.fallback import FallbackProvider

    return FallbackProvider(
        [
            get_provider(ProviderConfig(kind=ProviderKind.BEDROCK, model=m, region=region))
            for m in FALLBACK_MODELS
        ]
    )


def _is_hard(step) -> bool:  # noqa: ANN001
    name = (step.action.name or "").lower()
    return any(k in name for k in ("search", "list", "find")) or step.observation.is_error


def score(prompt, test, train, provider, embedder, max_tokens):  # noqa: ANN001, ANN201
    from wmh.engine.replay import replay
    from wmh.optimize.judge import RubricJudge
    from wmh.retrieval import EmbeddingRetriever

    # embedder=None -> zero-shot (no RAG): pass no retriever/train, matching evaluate_files.
    retriever = None
    if embedder is not None:
        retriever = EmbeddingRetriever(embedder)
        retriever.index(train)
    return replay(
        prompt,
        test,
        provider,
        RubricJudge(provider),
        retriever=retriever,
        train=train if embedder is not None else None,
        top_k=5,
        sample_turns="all",
        seed=0,
        max_tokens=max_tokens,
    )


def hard_mean(rep) -> tuple[float, int]:  # noqa: ANN001
    hs = [
        r.score
        for r in rep.results
        if any(k in (r.action or "").lower() for k in ("search", "list", "find"))
        or r.is_error_actual
    ]
    return (sum(hs) / len(hs) if hs else float("nan"), len(hs))


def run_one_candidate(idx, subset, val, base_prompt, provider, embedder, iterations):  # noqa: ANN001, ANN201
    """Optimize one GEPA candidate on a train `subset`. Returns (idx, evolved_prompt, changed)."""
    from wmh.optimize import GEPAOptimizer, RubricJudge
    from wmh.retrieval import EmbeddingRetriever

    opt = GEPAOptimizer(provider, RubricJudge(provider), retriever=EmbeddingRetriever(embedder))
    result = opt.optimize(
        subset, val, base_prompt, iterations, hard_step_filter=_is_hard, select_on_hard=True
    )
    return idx, result.prompt, result.prompt.strip() != base_prompt.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--train", type=float, default=0.5)
    ap.add_argument("--val", type=float, default=0.25)
    ap.add_argument("--test-cap", type=int, default=40)
    ap.add_argument("--region", default="us-west-1")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from wmh.engine.build import split_traces_3way
    from wmh.engine.prompts import BASE_ENV_PROMPT
    from wmh.ingest import get_adapter
    from wmh.retrieval import HashingEmbedder

    traces = get_adapter("otel-genai").from_file(args.file)
    train, val, test = split_traces_3way(traces, args.train, args.val)
    if args.test_cap and len(test) > args.test_cap:
        test = sorted(test, key=lambda t: t.trace_id)[: args.test_cap]
    print(
        f"{args.file}: train={len(train)} val={len(val)} test={len(test)} "
        f"(test steps={sum(len(t.steps) for t in test)})",
        flush=True,
    )

    provider = build_provider(args.region)
    embedder = HashingEmbedder(dim=512)

    # Each candidate optimizes on a different HALF of train (by trace-id bucketing), so they explore
    # different failure subsets; merge/knowledge-accumulation then compounds within each run.
    def subset(seed_mod: int) -> list:
        return [
            t for i, t in enumerate(sorted(train, key=lambda x: x.trace_id)) if i % 2 == seed_mod
        ] or train

    print(f"launching {args.candidates} GEPA candidates in parallel...", flush=True)
    results: list[tuple] = []
    with ThreadPoolExecutor(max_workers=args.candidates) as pool:
        futs = [
            pool.submit(
                run_one_candidate,
                i,
                subset(i % 2),
                val,
                BASE_ENV_PROMPT,
                provider,
                embedder,
                args.iterations,
            )
            for i in range(args.candidates)
        ]
        for f in futs:
            try:
                results.append(f.result())
            except Exception as exc:  # noqa: BLE001 - one candidate failing shouldn't sink the sweep
                print(f"  candidate failed: {exc}", flush=True)

    # Baselines on test.
    print("scoring baselines on test (base, base+RAG)...", flush=True)
    base_norag = score(BASE_ENV_PROMPT, test, train, provider, None, args.max_tokens)
    base_rag = score(BASE_ENV_PROMPT, test, train, provider, embedder, args.max_tokens)
    print(f"  base (no RAG)  = {base_norag.mean_score:.3f}", flush=True)
    print(f"  base + RAG     = {base_rag.mean_score:.3f}   <-- the number to beat", flush=True)

    # Each evolved candidate on test (with RAG).
    rows = []
    for idx, prompt, changed in results:
        rep = score(prompt, test, train, provider, embedder, args.max_tokens)
        hm, nh = hard_mean(rep)
        rows.append(
            {"idx": idx, "changed": changed, "test": rep.mean_score, "hard": hm, "prompt": prompt}
        )
        print(
            f"  candidate {idx}: test={rep.mean_score:.3f} hard={hm:.3f} "
            f"vs base+RAG {base_rag.mean_score:.3f} ({rep.mean_score - base_rag.mean_score:+.3f}) "
            f"changed={changed}",
            flush=True,
        )

    best = max(rows, key=lambda r: r["test"]) if rows else None
    bhm, nh = hard_mean(base_rag)
    if best:
        print(
            f"\n=== BEST GEPA vs base+RAG: {best['test'] - base_rag.mean_score:+.3f} "
            f"(overall {base_rag.mean_score:.3f} -> {best['test']:.3f}); "
            f"hard {bhm:.3f} -> {best['hard']:.3f} ===",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "file": args.file,
                "base_norag": base_norag.mean_score,
                "base_rag": base_rag.mean_score,
                "base_rag_hard": bhm,
                "candidates": [{k: v for k, v in r.items() if k != "prompt"} for r in rows],
                "best_prompt": best["prompt"] if best else None,
                "n_test_steps": base_rag.n_steps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
