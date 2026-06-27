"""Tests for benchmark run persistence + cross-seed aggregation."""

from __future__ import annotations

from wmh.bench.results import BenchRun, SeedResult, load_runs, save_run


def _seed(seed: int, mean: float, n_steps: int = 10, std: float = 0.0) -> SeedResult:
    return SeedResult(seed=seed, fidelity_mean=mean, fidelity_std=std, n_steps=n_steps)


def test_aggregate_step_weights_the_mean() -> None:
    run = BenchRun.aggregate(
        benchmark="tau",
        benchmark_version="1",
        prompt_label="base",
        sample_turns="all",
        rollouts=1,
        seeds=[_seed(0, 0.8, n_steps=10), _seed(1, 0.6, n_steps=30)],
    )
    # Step-weighted: (0.8*10 + 0.6*30) / 40 = 0.65, not the unweighted 0.7.
    assert run.fidelity_mean == 0.65
    assert run.total_steps == 40


def test_aggregate_reports_across_seed_std() -> None:
    run = BenchRun.aggregate(
        benchmark="tau",
        benchmark_version="1",
        prompt_label="base",
        sample_turns="all",
        rollouts=1,
        seeds=[_seed(0, 1.0), _seed(1, 0.0)],
    )
    # Population std of [1.0, 0.0] around mean 0.5 is 0.5.
    assert run.fidelity_std == 0.5


def test_aggregate_single_seed_has_zero_std() -> None:
    run = BenchRun.aggregate(
        benchmark="tau",
        benchmark_version="1",
        prompt_label="base",
        sample_turns="all",
        rollouts=1,
        seeds=[_seed(0, 0.9)],
    )
    assert run.fidelity_std == 0.0


def test_aggregate_no_steps_falls_back_to_seed_mean() -> None:
    run = BenchRun.aggregate(
        benchmark="tau",
        benchmark_version="1",
        prompt_label="base",
        sample_turns="all",
        rollouts=1,
        seeds=[_seed(0, 0.4, n_steps=0), _seed(1, 0.6, n_steps=0)],
    )
    assert run.fidelity_mean == 0.5  # unweighted when no steps were scored
    assert run.total_steps == 0


def test_save_then_load_round_trips(tmp_path) -> None:  # noqa: ANN001
    run = BenchRun.aggregate(
        benchmark="tau",
        benchmark_version="1",
        prompt_label="base",
        sample_turns="all",
        rollouts=2,
        seeds=[_seed(0, 0.8)],
        run_id="abc123",
        created_at="2026-06-26T00:00:00+00:00",
    )
    save_run(run, tmp_path / "results")
    loaded = load_runs(tmp_path / "results")
    assert len(loaded) == 1
    assert loaded[0].run_id == "abc123"
    assert loaded[0].fidelity_mean == 0.8
    assert loaded[0].seed_values == [0]


def test_load_missing_dir_is_empty(tmp_path) -> None:  # noqa: ANN001
    assert load_runs(tmp_path / "nope") == []
