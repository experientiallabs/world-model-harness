"""Leaderboard aggregation: persisted `BenchRun`s -> ranked rows.

Pure data shaping, no rich — the CLI (`wmh.cli.ui.leaderboard_table`) renders these rows into a
table. For each (benchmark, prompt) pair we surface the *latest* run (runs are comparable over time,
and the newest reflects the current prompt), then rank by fidelity. Keeping this headless makes the
ranking testable without a terminal.
"""

from __future__ import annotations

from pydantic import BaseModel

from wmh.bench.results import BenchRun


class LeaderboardRow(BaseModel):
    """One row: a prompt's latest score on a benchmark, with rollout/seed provenance to read it."""

    benchmark: str
    benchmark_version: str = "0"
    prompt_label: str
    fidelity_mean: float = 0.0
    fidelity_std: float = 0.0
    sample_turns: str = "all"
    rollouts: int = 1
    n_seeds: int = 0
    total_steps: int = 0
    run_id: str = ""
    created_at: str = ""


def build_leaderboard(runs: list[BenchRun]) -> list[LeaderboardRow]:
    """Collapse runs to the latest per (benchmark, prompt), sorted for display.

    Sort order: benchmark name, then fidelity (desc) so the best prompt leads each benchmark, then
    prompt label for a stable tie-break. "Latest" is decided by `created_at` then `run_id`, so an
    unstamped run never shadows a stamped one.
    """
    latest: dict[tuple[str, str], BenchRun] = {}
    for run in runs:
        key = (run.benchmark, run.prompt_label)
        current = latest.get(key)
        if current is None or _newer(run, current):
            latest[key] = run

    rows = [_row(run) for run in latest.values()]
    rows.sort(key=lambda r: (r.benchmark, -r.fidelity_mean, r.prompt_label))
    return rows


def _newer(candidate: BenchRun, current: BenchRun) -> bool:
    return (candidate.created_at, candidate.run_id) > (current.created_at, current.run_id)


def _row(run: BenchRun) -> LeaderboardRow:
    return LeaderboardRow(
        benchmark=run.benchmark,
        benchmark_version=run.benchmark_version,
        prompt_label=run.prompt_label,
        fidelity_mean=run.fidelity_mean,
        fidelity_std=run.fidelity_std,
        sample_turns=run.sample_turns,
        rollouts=run.rollouts,
        n_seeds=len(run.seeds),
        total_steps=run.total_steps,
        run_id=run.run_id,
        created_at=run.created_at,
    )


__all__ = ["LeaderboardRow", "build_leaderboard"]
