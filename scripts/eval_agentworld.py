#!/usr/bin/env python
"""Baseline #5: Qwen-AgentWorld-35B as the world model, scored by the same Opus rubric judge.

The world model runs on the H100 box's vLLM server (OpenAI-compatible, served at
`OPENAI_BASE_URL`); the judge is Bedrock Opus 4.8 so the fidelity number is directly comparable to
the Opus baselines. `wmh eval` couples both to one provider, so this small runner drives
`evaluate_files` with the two providers split apart.

    OPENAI_BASE_URL=http://localhost:8001/v1 OPENAI_API_KEY=dummy AWS_REGION=us-west-1 \
      uv run python scripts/eval_agentworld.py examples/tau2-bench.otel.jsonl \
        --aw-model Qwen/Qwen-AgentWorld-35B-A3B --region us-west-1 \
        --train-split 0.7 --seed 0 --out benchmarks/results/grid-agentworld-rag.json

AgentWorld is a reasoning model (long traces, ~17 tok/s, max-num-seqs 1), so each step can take
minutes; the run is intentionally serial and patient. RAG is on by default (top-k=5, leak-free) to
match baseline #4's retrieval condition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--aw-model", default="Qwen/Qwen-AgentWorld-35B-A3B")
    ap.add_argument("--region", default="us-west-1", help="Bedrock region for the judge.")
    ap.add_argument("--judge-model", default="us.anthropic.claude-opus-4-8")
    ap.add_argument("--train-split", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embed-dim", type=int, default=512)
    ap.add_argument("--no-rag", action="store_true", help="Disable retrieval (zero-shot replay).")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from wmh.engine.eval import evaluate_files
    from wmh.engine.prompts import BASE_ENV_PROMPT
    from wmh.optimize.judge import RubricJudge
    from wmh.providers import ProviderConfig, ProviderKind, get_provider
    from wmh.retrieval import HashingEmbedder

    # World model: AgentWorld via the OpenAI-compatible vLLM endpoint (OPENAI_BASE_URL in env).
    world = get_provider(ProviderConfig(kind=ProviderKind.OPENAI, model=args.aw_model))
    # Judge: Bedrock Opus 4.8 — the SAME scorer as every other baseline, for comparability.
    judge_llm = get_provider(
        ProviderConfig(kind=ProviderKind.BEDROCK, model=args.judge_model, region=args.region)
    )
    embedder = None if args.no_rag else HashingEmbedder(dim=args.embed_dim)

    report = evaluate_files(
        [Path(f) for f in args.files],
        BASE_ENV_PROMPT,  # AgentWorld is a trained world model; it gets the plain base prompt.
        world,
        RubricJudge(judge_llm),
        embedder=embedder,
        train_split=args.train_split,
        sample_turns="all",
        seed=args.seed,
    )
    for name, rep in report.per_file.items():
        print(f"  {name:28} {rep.summary()}")
    print(
        f"OVERALL fidelity={report.overall_fidelity:.3f}±{report.overall_std:.3f} "
        f"over {report.total_steps} held-out steps"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({n: r.model_dump() for n, r in report.per_file.items()}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote report -> {args.out}")


if __name__ == "__main__":
    main()
