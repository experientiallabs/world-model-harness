"""Run reconstruction-fidelity replay across the examples/ corpus and print a scorecard.

Loads each examples/<benchmark>.otel.jsonl, splits train/holdout deterministically, replays the
held-out steps through a world-model prompt (BASE_ENV_PROMPT by default, or --prompt <file>), scores
predicted vs. actual observations with the LLM judge, and reports per-benchmark + overall fidelity.

Uses Bedrock Opus 4.8 by default (the live backend here). Requires AWS_REGION. Example:

    AWS_REGION=us-east-1 uv run python scripts/replay_eval.py --benchmarks tau2-bench,bird-sql

This is the loop behind iterating on BASE_ENV_PROMPT (docs/base_prompt_iteration.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wmh.engine.build import split_traces
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.engine.replay import ReplayReport, replay
from wmh.ingest import get_adapter
from wmh.optimize.judge import LLMJudge
from wmh.providers import ProviderConfig, ProviderKind, get_provider
from wmh.retrieval import EmbeddingRetriever, HashingEmbedder

_EXAMPLES = Path("examples")


def run(
    benchmarks: list[str],
    prompt: str,
    *,
    model: str,
    region: str,
    train_split: float,
    use_rag: bool,
    embed_dim: int,
) -> dict[str, ReplayReport]:
    provider = get_provider(
        ProviderConfig(kind=ProviderKind.BEDROCK, model=model, region=region)
    )
    judge = LLMJudge(provider)
    adapter = get_adapter("otel-genai")
    reports: dict[str, ReplayReport] = {}
    for benchmark in benchmarks:
        path = _EXAMPLES / f"{benchmark}.otel.jsonl"
        if not path.exists():
            print(f"  (skip {benchmark}: no {path})")
            continue
        traces = adapter.from_file(str(path))
        train, holdout = split_traces(traces, train_split)
        if not holdout:  # tiny corpus: evaluate on everything
            train, holdout = traces, traces
        retriever = EmbeddingRetriever(HashingEmbedder(dim=embed_dim)) if use_rag else None
        report = replay(
            prompt, holdout, provider, judge, retriever=retriever, train=train if use_rag else None
        )
        reports[benchmark] = report
        print(f"  {benchmark:28} {report.summary()}")
    return reports


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", default="", help="Comma-separated; default = all examples.")
    parser.add_argument("--prompt", default="", help="Prompt file path; default = BASE_ENV_PROMPT.")
    parser.add_argument("--model", default="us.anthropic.claude-opus-4-8")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--no-rag", action="store_true", help="Disable retrieval (zero-shot).")
    parser.add_argument("--out", default="", help="Optional path for the full JSON report.")
    args = parser.parse_args(argv[1:])

    benchmarks = (
        [b.strip() for b in args.benchmarks.split(",") if b.strip()]
        if args.benchmarks
        else sorted(p.stem.removesuffix(".otel") for p in _EXAMPLES.glob("*.otel.jsonl"))
    )
    prompt = Path(args.prompt).read_text(encoding="utf-8") if args.prompt else BASE_ENV_PROMPT

    reports = run(
        benchmarks,
        prompt,
        model=args.model,
        region=args.region,
        train_split=args.train_split,
        use_rag=not args.no_rag,
        embed_dim=args.embed_dim,
    )
    if reports:
        overall = sum(r.mean_score * r.n_steps for r in reports.values())
        n = sum(r.n_steps for r in reports.values())
        print(f"\nOVERALL fidelity={overall / n:.3f} over {n} held-out steps")
    if args.out:
        Path(args.out).write_text(
            json.dumps({b: r.model_dump() for b, r in reports.items()}, indent=2), encoding="utf-8"
        )
        print(f"wrote full report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
