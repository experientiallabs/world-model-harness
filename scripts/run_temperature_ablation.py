#!/usr/bin/env python
"""Live runner for the train-vs-eval temperature ablation (the harness's first experiment).

This is the SIDECAR: it imports and wraps the pipeline (`wmh.research`) rather than reimplementing
it, so it measures what the harness actually does. It ingests a trace file, splits it the same way
`wmh build` does, then sweeps the (T_train × T_eval) grid across multiple seeds on a live provider,
printing per-cell mean ± std and writing the full `AblationReport` JSON.

    AWS_REGION=us-east-1 uv run python scripts/run_temperature_ablation.py \
        examples/terminal-bench.otel.jsonl \
        --seeds 0,1,2 --budget 12 --out report.json

Defaults to Bedrock Opus 4.8 with the offline HashingEmbedder (no embedding creds), matching the
canonical build. Use `--temps` to change the grid (e.g. `0,0.5,1`). See docs/gepa_research.md.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from wmh.engine.build import split_traces
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.ingest import get_adapter
from wmh.optimize.judge import Judge, LLMJudge
from wmh.providers import ProviderConfig, ProviderKind, get_provider
from wmh.providers.base import Embedder, Provider
from wmh.research import TemperatureAblation, run_ablation, temperature_conditions
from wmh.research.ablation import AblationReport, Condition
from wmh.retrieval import HashingEmbedder


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _make_backends(
    provider: ProviderKind, model: str, region: str | None, embed_dim: int, no_rag: bool
) -> Callable[[], tuple[Provider, Judge, Embedder | None]]:
    """A factory the ablation calls once per run to get (provider, judge, embedder).

    The provider/judge are shared LLM clients (stateless across runs); the embedder is the offline
    HashingEmbedder (no creds) or None for zero-shot. Built once and reused so every cell hits the
    same backend.
    """
    llm: Provider = get_provider(ProviderConfig(kind=provider, model=model, region=region))
    judge: Judge = LLMJudge(llm)
    embedder: Embedder | None = None if no_rag else HashingEmbedder(dim=embed_dim)

    def factory() -> tuple[Provider, Judge, Embedder | None]:
        return llm, judge, embedder

    return factory


def _run(args: argparse.Namespace) -> AblationReport:
    traces = get_adapter("otel-genai").from_file(args.file)
    if not traces:
        raise SystemExit(f"no traces ingested from {args.file}")
    train, held_out = split_traces(traces, args.train_split)
    if not held_out:  # tiny corpus: score on everything (same fallback as `wmh eval`)
        train, held_out = traces, traces

    n_train = sum(len(t.steps) for t in train)
    n_held = sum(len(t.steps) for t in held_out)
    print(
        f"corpus {Path(args.file).name}: {len(traces)} traces, "
        f"train={n_train} steps, held-out={n_held} steps"
    )
    temps = _parse_floats(args.temps)
    seeds = _parse_ints(args.seeds)
    print(f"grid: T in {temps} (={len(temps) ** 2} cells), seeds={seeds}, budget={args.budget}\n")

    ablation = TemperatureAblation(
        train,
        held_out,
        BASE_ENV_PROMPT,
        make_backends=_make_backends(
            ProviderKind(args.provider), args.model, args.region, args.embed_dim, args.no_rag
        ),
        budget=args.budget,
        conditions=temperature_conditions(temps),
    )

    def _progress(condition: Condition, seed: int, score: float) -> None:
        print(f"  {condition.label:20} seed={seed}  fidelity={score:.3f}")

    return run_ablation(ablation, seeds, on_run=_progress)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="OTel trace file to build/eval against.")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated GEPA seeds.")
    parser.add_argument("--temps", default="0,1", help="Comma-separated temperatures for the grid.")
    parser.add_argument("--budget", type=int, default=12, help="GEPA rollout budget per run.")
    parser.add_argument("--train-split", type=float, default=0.7, help="Train/holdout ratio.")
    parser.add_argument("--provider", default="bedrock", help="Provider kind.")
    parser.add_argument("--model", default="us.anthropic.claude-opus-4-8", help="Model id.")
    parser.add_argument("--region", default="us-east-1", help="AWS region (Bedrock).")
    parser.add_argument("--embed-dim", type=int, default=512, help="phi dim (offline embedder).")
    parser.add_argument("--no-rag", action="store_true", help="Disable retrieval (zero-shot).")
    parser.add_argument("--out", default=None, help="Path to write the full AblationReport JSON.")
    args = parser.parse_args()

    report = _run(args)

    print(f"\n=== {report.name} (seeds={report.seeds}) ===")
    for cell in report.conditions:
        print(f"  {cell.summary()}")
    best = report.best()
    if best is not None:
        print(f"  best: {best.condition.label} (mean fidelity {best.mean:.3f})")

    if args.out:
        Path(args.out).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nwrote report -> {args.out}")


if __name__ == "__main__":
    main()
