"""Tests for the benchmark runner: one scoring pass per seed, aggregated into a BenchRun."""

from __future__ import annotations

from pathlib import Path

import pytest

from wmh.bench.definition import BenchmarkDef, EvalConfig
from wmh.bench.runner import RolloutScore, run_benchmark


def _bench(tmp_path, seeds: list[int]) -> BenchmarkDef:  # noqa: ANN001 - pytest path
    return BenchmarkDef(
        name="tau",
        version="1",
        traces=["a.jsonl"],
        eval=EvalConfig(seeds=seeds, rollouts=2, temperature=0.7),
        dir=tmp_path,
    )


def test_runs_one_score_pass_per_seed() -> None:
    seen: list[int] = []

    def score_once(files, prompt, *, sample_turns, rollouts, temperature, seed):  # noqa: ANN001, ANN202
        seen.append(seed)
        # Echo the knobs back so we can assert they flow through from the config.
        assert rollouts == 2
        assert temperature == 0.7
        assert sample_turns == "all"
        return RolloutScore(fidelity_mean=0.8, fidelity_std=0.1, n_steps=10, rollouts=rollouts)

    bench = _bench(Path("/tmp"), seeds=[0, 1, 7])
    run = run_benchmark(bench, "PROMPT", "base", score_once)

    assert seen == [0, 1, 7]
    assert run.benchmark == "tau"
    assert run.prompt_label == "base"
    assert run.rollouts == 2
    assert run.fidelity_mean == 0.8  # all seeds equal -> mean 0.8
    assert run.fidelity_std == pytest.approx(0.0)  # no spread across seeds
    assert run.total_steps == 30
    assert run.seed_values == [0, 1, 7]


def test_aggregates_across_seed_variance() -> None:
    scores = {0: 1.0, 1: 0.0}

    def score_once(files, prompt, *, sample_turns, rollouts, temperature, seed):  # noqa: ANN001, ANN202
        return RolloutScore(fidelity_mean=scores[seed], n_steps=10, rollouts=rollouts)

    bench = _bench(Path("/tmp"), seeds=[0, 1])
    run = run_benchmark(bench, "PROMPT", "base", score_once)
    assert run.fidelity_mean == 0.5
    assert run.fidelity_std == 0.5  # population std of [1.0, 0.0]


def test_on_seed_callback_fires_per_seed() -> None:
    progress: list[tuple[int, int, int]] = []

    def score_once(files, prompt, *, sample_turns, rollouts, temperature, seed):  # noqa: ANN001, ANN202
        return RolloutScore(fidelity_mean=0.5, n_steps=5, rollouts=rollouts)

    bench = _bench(Path("/tmp"), seeds=[3, 4])
    run_benchmark(
        bench, "PROMPT", "base", score_once, on_seed=lambda s, d, t: progress.append((s, d, t))
    )
    assert progress == [(3, 1, 2), (4, 2, 2)]
