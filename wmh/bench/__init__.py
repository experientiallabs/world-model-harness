"""Benchmarks as first-class, persisted objects on top of the open-loop eval scorer.

A *benchmark* is a committed, versioned definition (`benchmarks/<name>/benchmark.toml`): a named set
of recorded trace files plus an eval config (sample-turns, rollouts, seeds, judge). `wmh bench run`
scores a world-model prompt against it reproducibly — per seed, aggregating the rollout distribution
as MEAN + STD — and persists the result under `benchmarks/<name>/results/` ("filesystem as DB").
`wmh bench` / `wmh bench list` render a leaderboard over those persisted runs.

The scoring unit itself (replay a held-out trace teacher-forced, judge predicted vs. real
observation) lives in the eval layer (`wmh.engine.eval`); this package never reimplements it — it
calls it once per rollout and aggregates.
"""

from wmh.bench.definition import (
    BenchmarkDef,
    EvalConfig,
    JudgeConfig,
    discover_benchmarks,
    load_benchmark,
)
from wmh.bench.leaderboard import LeaderboardRow, build_leaderboard
from wmh.bench.results import (
    BenchRun,
    SeedResult,
    load_runs,
    results_dir_for,
    save_run,
)
from wmh.bench.runner import RolloutScore, ScoreOnce, run_benchmark
from wmh.bench.scoring import evaluate_files_once

__all__ = [
    "BenchmarkDef",
    "EvalConfig",
    "JudgeConfig",
    "discover_benchmarks",
    "load_benchmark",
    "LeaderboardRow",
    "build_leaderboard",
    "BenchRun",
    "SeedResult",
    "load_runs",
    "results_dir_for",
    "save_run",
    "RolloutScore",
    "ScoreOnce",
    "run_benchmark",
    "evaluate_files_once",
]
