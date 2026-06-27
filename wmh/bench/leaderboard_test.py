"""Tests for leaderboard aggregation over persisted runs."""

from __future__ import annotations

from wmh.bench.leaderboard import build_leaderboard
from wmh.bench.results import BenchRun, SeedResult


def _run(
    benchmark: str,
    prompt: str,
    mean: float,
    *,
    run_id: str,
    created_at: str = "",
) -> BenchRun:
    return BenchRun.aggregate(
        benchmark=benchmark,
        benchmark_version="1",
        prompt_label=prompt,
        sample_turns="all",
        rollouts=1,
        seeds=[SeedResult(seed=0, fidelity_mean=mean, n_steps=10)],
        run_id=run_id,
        created_at=created_at,
    )


def test_ranks_prompts_by_fidelity_within_benchmark() -> None:
    rows = build_leaderboard(
        [
            _run("tau", "base", 0.5, run_id="r1"),
            _run("tau", "optimized", 0.9, run_id="r2"),
        ]
    )
    assert [r.prompt_label for r in rows] == ["optimized", "base"]


def test_keeps_only_latest_run_per_prompt() -> None:
    rows = build_leaderboard(
        [
            _run("tau", "base", 0.5, run_id="old", created_at="2026-01-01T00:00:00+00:00"),
            _run("tau", "base", 0.8, run_id="new", created_at="2026-06-01T00:00:00+00:00"),
        ]
    )
    assert len(rows) == 1
    assert rows[0].run_id == "new"
    assert rows[0].fidelity_mean == 0.8


def test_groups_by_benchmark_then_fidelity() -> None:
    rows = build_leaderboard(
        [
            _run("retail", "base", 0.7, run_id="r1"),
            _run("tau", "base", 0.6, run_id="r2"),
            _run("tau", "opt", 0.95, run_id="r3"),
        ]
    )
    # Sorted by benchmark name, then fidelity desc.
    assert [(r.benchmark, r.prompt_label) for r in rows] == [
        ("retail", "base"),
        ("tau", "opt"),
        ("tau", "base"),
    ]


def test_empty_runs_yield_no_rows() -> None:
    assert build_leaderboard([]) == []
