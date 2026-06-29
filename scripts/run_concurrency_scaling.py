#!/usr/bin/env python
"""Live runner for the concurrency scaling law: batch wall-clock vs. how many scenarios run at once.

The SIDECAR for `wmh.research.run_concurrency_scaling` — it loads a benchmark's bundled world model
and corpus, picks a fixed batch of N held-out scenarios, and times reconstructing that batch at each
concurrency level (1, 2, 4, …). With `--side both` it ALSO times the matching real sandbox batch at
each level (via `tools/<bench>-capture/run.sh`), so the report carries the *time differential*
T_real(W) / T_world(W) — the world model's standup-free advantage as concurrency rises.

    # world-model side only (no tau2 venv needed): the cleanest speedup curve
    AWS_REGION=us-east-1 uv run python scripts/run_concurrency_scaling.py tau-bench \
        --scenarios 8 --concurrency-levels 1,2,4,8 --side world --out conc_world.json

    # both sides (needs tools/tau2-capture venv + TAU2_DATA_DIR): the differential
    AWS_REGION=us-east-1 uv run python scripts/run_concurrency_scaling.py tau-bench \
        --scenarios 8 --concurrency-levels 1,2,4,8 --side both --out conc_both.json

Reuses the deployed primitives: `wmh.bench.run_scenario` (world side, own provider + tracker per
scenario) and `wmh.bench.side_by_side.run_real_sandbox` (real side), fanned across a
`ThreadPoolExecutor` exactly like `wmh bench side-by-side`. tau-bench today; terminal-tasks and
swe-bench are just a different benchmark name. See docs/concurrency_scaling.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import time
from pathlib import Path

from wmh.bench import (
    ScenarioReport,
    load_benchmark,
    run_scenario,
    select_scenarios,
)
from wmh.bench.side_by_side import real_sandbox_spec, run_real_sandbox, runner_info
from wmh.config import ArtifactPaths, WorldModelStore, load_config
from wmh.core.types import Trace
from wmh.engine.build import split_traces
from wmh.engine.prompts import BASE_ENV_PROMPT
from wmh.ingest import get_adapter
from wmh.providers import get_provider
from wmh.providers.base import ProviderConfig
from wmh.research import (
    ConcurrencyScalingReport,
    RealBatch,
    Side,
    WorldBatch,
    run_concurrency_scaling,
)
from wmh.research.concurrency_scaling import ConcurrencyPoint, RealRunner, WorldRunner
from wmh.retrieval import EmbeddingRetriever, get_embedder
from wmh.retrieval.leakfree import DemoRetriever

DEFAULT_LEVELS = "1,2,4,8"


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _build_world_runner(
    *,
    serve_config: ProviderConfig,
    env_prompt: str,
    name: str,
    model_label: str,
    demos: DemoRetriever,
    selected: list[tuple[int, Trace]],
) -> WorldRunner:
    """A `WorldRunner`: run the N selected scenarios at concurrency `level`, return a `WorldBatch`.

    Each scenario gets its OWN provider (boto3 clients aren't safe to share across threads) and its
    own `RunTracker` inside `run_scenario`, so concurrent batches never race on metering — the same
    safety the merged `wmh bench side-by-side` relies on. The leak-free `demos` index is built once
    and shared read-only (query path is stateless), so index-build time never pollutes the timing.
    """

    def run_one(scenario_trace: Trace) -> ScenarioReport:
        provider = get_provider(serve_config)
        return run_scenario(
            provider,
            env_prompt,
            scenario_trace,
            demos,
            benchmark=name,
            model=model_label,
        )

    def runner(level: int) -> WorldBatch:
        start = time.monotonic()
        reports: list[ScenarioReport] = []
        with cf.ThreadPoolExecutor(max_workers=level) as executor:
            futures = [executor.submit(run_one, t) for _idx, t in selected]
            for future in cf.as_completed(futures):
                reports.append(future.result())
        wall = time.monotonic() - start
        steps = sum(len(r.steps) for r in reports)
        matches = sum(
            1
            for r in reports
            for s in r.steps
            if s.is_error_predicted == s.is_error_actual
        )
        return WorldBatch(
            wall_seconds=wall,
            work_seconds=sum(r.total_seconds for r in reports),
            ok=len(reports),
            total=len(selected),
            tokens=sum(r.tokens for r in reports),
            cost_usd=sum(r.cost_usd for r in reports),
            fidelity=(matches / steps) if steps else 0.0,
        )

    return runner


def _build_real_runner(
    *,
    name: str,
    train_split: float,
    selected: list[tuple[int, Trace]],
    trace_pool: str | None,
    extra_args: list[str],
    timeout: float | None,
) -> RealRunner:
    """A `RealRunner`: run the N matching real sandboxes at concurrency `level` -> a `RealBatch`."""

    def run_one(pool_index: int) -> float:
        spec = real_sandbox_spec(
            name,
            trace_index=pool_index,
            train_split=train_split,
            trace_pool=trace_pool,
            extra_args=extra_args,
        )
        result = run_real_sandbox(spec, timeout_seconds=timeout)
        return result.seconds if result.ok else -result.seconds

    def runner(level: int) -> RealBatch:
        start = time.monotonic()
        seconds: list[float] = []
        with cf.ThreadPoolExecutor(max_workers=level) as executor:
            futures = [executor.submit(run_one, idx) for idx, _t in selected]
            for future in cf.as_completed(futures):
                seconds.append(future.result())
        wall = time.monotonic() - start
        ok = sum(1 for s in seconds if s >= 0)
        return RealBatch(
            wall_seconds=wall,
            work_seconds=sum(abs(s) for s in seconds),
            ok=ok,
            total=len(selected),
        )

    return runner


def _run(args: argparse.Namespace) -> ConcurrencyScalingReport:
    side = Side(args.side)
    levels = _parse_ints(args.levels)
    if not levels:
        raise SystemExit("--concurrency-levels must list at least one level")

    bench = load_benchmark(Path(args.benchmarks) / args.benchmark)
    missing = bench.missing_traces()
    if missing:
        raise SystemExit(f"benchmark {args.benchmark!r} missing traces: {missing}")
    if side in (Side.BOTH, Side.REAL):
        runner_info(args.benchmark)  # fail early if there's no real runner

    adapter = get_adapter("otel-genai")
    traces = [t for f in bench.trace_files() for t in adapter.from_file(str(f))]
    if not traces:
        raise SystemExit(f"benchmark {args.benchmark!r} ingested no traces")
    train, holdout = split_traces(traces, bench.eval.train_split)
    selection = select_scenarios(traces, holdout, trace_index=None, scenarios=args.scenarios)
    selected = selection.scenarios

    # Load the bundled world model (prompt + serve provider + embedder), like the side-by-side CLI.
    store = WorldModelStore(args.root)
    model_dir = store.resolve(args.model or args.benchmark)
    config = load_config(str(model_dir))
    paths = ArtifactPaths(model_dir)
    env_prompt = (
        paths.optimized_prompt.read_text(encoding="utf-8")
        if paths.optimized_prompt.exists()
        else BASE_ENV_PROMPT
    )
    serve_config = config.serve_provider_config()
    if args.serve_model:
        serve_config = serve_config.model_copy(update={"model": args.serve_model})

    # Build the leak-free demo index ONCE (shared read-only across workers); train-only, never the
    # query's own trace — identical to eval.
    retriever = EmbeddingRetriever(get_embedder(config))
    demos = DemoRetriever(retriever, train or traces, top_k=config.top_k)

    world_runner = _build_world_runner(
        serve_config=serve_config,
        env_prompt=env_prompt,
        name=args.benchmark,
        model_label=args.model or args.benchmark,
        demos=demos,
        selected=selected,
    )

    real_runner = None
    if side in (Side.BOTH, Side.REAL):
        extra_args = list(args.real_arg or [])
        info = runner_info(args.benchmark)
        # Concurrent cold runs of some runners purge a shared image family; force warm+cache so a
        # multi-scenario batch is safe (mirrors `wmh bench side-by-side`).
        if info.concurrent_purges_images and args.scenarios > 1:
            for flag in ("--warm", "--cache"):
                if flag not in extra_args:
                    extra_args.append(flag)
        real_runner = _build_real_runner(
            name=args.benchmark,
            train_split=bench.eval.train_split,
            selected=selected,
            trace_pool="all" if selection.widened else None,
            extra_args=extra_args,
            timeout=args.real_timeout,
        )

    print(
        f"benchmark {args.benchmark}: {len(traces)} traces, batch of {len(selected)} "
        f"{selection.pool_kind} scenario(s), levels={levels}, side={side.value}, "
        f"trials={args.trials}\n"
    )

    def _progress(point: ConcurrencyPoint) -> None:
        print(f"  {point.summary()}")

    report = run_concurrency_scaling(
        world_runner,
        real_runner,
        levels=levels,
        scenarios=len(selected),
        trials=args.trials,
        side=side,
        on_point=_progress,
    )
    report.benchmark = args.benchmark
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", help="Benchmark name (benchmarks/<name>/benchmark.toml).")
    parser.add_argument("--scenarios", type=int, default=8, help="Batch size N held fixed.")
    parser.add_argument(
        "--concurrency-levels", dest="levels", default=DEFAULT_LEVELS,
        help="Comma-separated concurrency levels to sweep (put the baseline first).",
    )
    parser.add_argument(
        "--trials", type=int, default=1, help="Timed repeats per level (for error bars)."
    )
    parser.add_argument(
        "--side", default="both", choices=[s.value for s in Side],
        help="both = differential (default); world = cheap WM-only; real = sandbox-only.",
    )
    parser.add_argument("--model", default=None, help="World model dir (default: benchmark name).")
    parser.add_argument(
        "--serve-model", default=None, help="Override the LLM that plays the environment."
    )
    parser.add_argument(
        "--real-arg", action="append", default=None,
        help="Extra arg forwarded to the real sandbox runner; repeat, e.g. --real-arg=--cache.",
    )
    parser.add_argument(
        "--real-timeout", type=float, default=None, help="Abort a real sandbox run after N seconds."
    )
    parser.add_argument("--benchmarks", default="benchmarks", help="Benchmark defs dir.")
    parser.add_argument("--root", default=".wmh", help="Project dir (for --model).")
    parser.add_argument(
        "--out", default=None, help="Path to write the ConcurrencyScalingReport JSON."
    )
    args = parser.parse_args()

    report = _run(args)

    print(f"\n=== {report.name}: {report.benchmark or args.benchmark} (N={report.scenarios}) ===")
    for point in report.points:
        print(f"  {point.summary()}")
    best = report.best_speedup()
    if best is not None and best.speedup:
        print(
            f"  best world-model speedup: {best.speedup:.2f}x at concurrency {best.level} "
            f"(efficiency {best.efficiency:.0%})"
        )

    if args.out:
        Path(args.out).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nwrote report -> {args.out}")


if __name__ == "__main__":
    main()
