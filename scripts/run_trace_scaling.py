#!/usr/bin/env python
"""Live runner for the trace scaling law: fidelity vs. number of training traces.

The SIDECAR for `wmh.research.TraceScalingAblation` — it ingests a corpus, holds a fixed test/valid
split, and sweeps the TRAIN trace count (e.g. 10, 20, 50, 100, … capped at the corpus) for one or
both modes (`base` = shipped prompt + RAG, `gepa` = GEPA-optimized per count), reporting test
fidelity mean ± std across seeds at each point. The curve says whether more traces keep buying
fidelity or saturate — and for tau2 we expect earlier saturation than a richer domain.

    # tau2 today (66 traces): cheap base curve first, then the GEPA curve
    AWS_REGION=us-east-1 uv run python scripts/run_trace_scaling.py tau-bench \
        --counts 10,20,40 --modes base --seeds 0,1 --out scaling_base.json
    AWS_REGION=us-east-1 uv run python scripts/run_trace_scaling.py tau-bench \
        --counts 10,20,40 --modes gepa --budget 12 --seeds 0,1 --out scaling_gepa.json

Accepts a benchmark NAME (resolved from `benchmarks/<name>/benchmark.toml` — reusing its corpus +
pinned judge) or a raw OTel `--file`. Defaults to Bedrock Opus 4.8 + offline HashingEmbedder, the
canonical build. To push counts toward 1000, grow the corpus first (see
docs/trace_scaling.md / tools/tau2-capture). See docs/gepa_research.md for the framework.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from wmh.bench.definition import BenchmarkDef, load_benchmark
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.ingest import get_adapter
from wmh.optimize.judge import Judge, LLMJudge, RubricJudge
from wmh.providers import ProviderConfig, ProviderKind, get_provider
from wmh.providers.base import Embedder, Provider
from wmh.research import TraceScalingAblation, run_ablation
from wmh.research.ablation import AblationReport, Condition
from wmh.retrieval import HashingEmbedder

# Default scaling ladder: dense at the low end (where the curve bends), capped at the corpus by the
# ablation. Override with --counts.
DEFAULT_COUNTS = "10,20,50,100,200,500,1000"


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_modes(text: str) -> list[str]:
    return [m.strip() for m in text.split(",") if m.strip()]


def _make_backends(
    provider: ProviderKind, model: str, region: str | None, embed_dim: int, no_rag: bool, judge: str
) -> Callable[[], tuple[Provider, Judge, Embedder | None]]:
    """Factory the ablation calls per run for (provider, judge, embedder). Cf. seed-stability."""
    llm: Provider = get_provider(ProviderConfig(kind=provider, model=model, region=region))
    scorer: Judge = RubricJudge(llm) if judge == "rubric" else LLMJudge(llm)
    embedder: Embedder | None = None if no_rag else HashingEmbedder(dim=embed_dim)

    def factory() -> tuple[Provider, Judge, Embedder | None]:
        return llm, scorer, embedder

    return factory


def _load_corpus(args: argparse.Namespace) -> tuple[list, str]:  # noqa: ANN201 - (traces, label)
    """Resolve the corpus from a benchmark name (preferred) or a raw --file -> (traces, label)."""
    adapter = get_adapter("otel-genai")
    if args.benchmark:
        bench: BenchmarkDef = load_benchmark(Path(args.benchmarks) / args.benchmark)
        missing = bench.missing_traces()
        if missing:
            raise SystemExit(f"benchmark {args.benchmark!r} missing traces: {missing}")
        traces = [t for f in bench.trace_files() for t in adapter.from_file(str(f))]
        return traces, args.benchmark
    if not args.file:
        raise SystemExit("pass a benchmark name or --file <trace.jsonl>")
    return adapter.from_file(args.file), Path(args.file).name


def _run(args: argparse.Namespace) -> AblationReport:
    traces, label = _load_corpus(args)
    if not traces:
        raise SystemExit("no traces ingested")

    seeds = _parse_ints(args.seeds)
    ablation = TraceScalingAblation(
        traces,
        BASE_ENV_PROMPT,
        make_backends=_make_backends(
            ProviderKind(args.provider),
            args.model,
            args.region,
            args.embed_dim,
            args.no_rag,
            args.judge,
        ),
        counts=_parse_ints(args.counts),
        modes=_parse_modes(args.modes),
        budget=args.budget,
        top_k=args.top_k,
        test_frac=args.test_frac,
        valid_frac=args.valid_frac,
    )
    split = ablation.split
    print(
        f"corpus {label}: {len(traces)} traces -> "
        f"train pool {len(split.train_pool)}, valid {len(split.valid)}, test {len(split.test)}"
    )
    print(f"counts={ablation.counts}, modes={args.modes}, seeds={seeds}, budget={args.budget}\n")

    def _progress(condition: Condition, seed: int, score: float) -> None:
        print(f"  {condition.label:14} seed={seed}  fidelity={score:.3f}")

    return run_ablation(ablation, seeds, on_run=_progress)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", nargs="?", help="Benchmark name (benchmarks/<name>/).")
    parser.add_argument("--file", default=None, help="Raw OTel trace file (not a benchmark).")
    parser.add_argument("--benchmarks", default="benchmarks", help="Benchmark definitions dir.")
    parser.add_argument("--counts", default=DEFAULT_COUNTS, help="Comma-separated train counts.")
    parser.add_argument("--modes", default="base,gepa", help="Comma-separated: base, gepa.")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds (error bars).")
    parser.add_argument("--budget", type=int, default=12, help="GEPA rollout budget (gepa mode).")
    parser.add_argument("--top-k", type=int, default=5, help="Retrieval depth.")
    parser.add_argument("--test-frac", type=float, default=0.2, help="Fixed test fraction.")
    parser.add_argument("--valid-frac", type=float, default=0.15, help="Fixed valid fraction.")
    parser.add_argument("--provider", default="bedrock", help="Provider kind.")
    parser.add_argument("--model", default="us.anthropic.claude-opus-4-8", help="Model id.")
    parser.add_argument("--region", default="us-east-1", help="AWS region (Bedrock).")
    parser.add_argument("--embed-dim", type=int, default=512, help="phi dim (offline embedder).")
    parser.add_argument("--no-rag", action="store_true", help="Disable retrieval (zero-shot).")
    parser.add_argument("--judge", default="rubric", help="Scorer: rubric (5-dim) | match.")
    parser.add_argument("--out", default=None, help="Path to write the AblationReport JSON.")
    args = parser.parse_args()

    report = _run(args)

    print(f"\n=== {report.name} (seeds={report.seeds}) ===")
    for cell in report.conditions:
        print(f"  {cell.summary()}")
    if args.out:
        Path(args.out).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nwrote report -> {args.out}")


if __name__ == "__main__":
    main()
