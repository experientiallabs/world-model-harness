"""Persisted benchmark runs ("filesystem as DB"): one JSON per run in `benchmarks/<name>/results`.

A `BenchRun` is the comparable-over-time record of scoring one prompt against one benchmark: the
per-seed fidelities (each already a rollout mean + std) and the across-seed aggregate. Runs persist
so the leaderboard can compare prompts and seeds historically without rerunning. The layout mirrors
`wmh.tracking.store` (one JSON file per record, dependency-light) so it reads like the rest of the
harness.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from wmh.bench._stats import pop_std

RESULTS_DIRNAME = "results"


class SeedResult(BaseModel):
    """One seed's fidelity, aggregated over that seed's world-model rollouts."""

    seed: int
    fidelity_mean: float = 0.0
    fidelity_std: float = 0.0  # std across the seed's rollouts
    rollouts: int = 1
    n_steps: int = 0


class BenchRun(BaseModel):
    """One persisted scoring of a prompt against a benchmark, aggregated across seeds.

    `fidelity_mean` is the step-weighted mean of the per-seed means; `fidelity_std` is the
    (population) std across seed means — a reproducibility signal distinct from within-seed rollout
    std. `run_id` and `created_at` are stamped by the caller (no implicit clock here) so runs sort
    and dedupe deterministically.
    """

    run_id: str
    created_at: str = ""  # ISO-8601, stamped by the caller; "" if unknown
    benchmark: str
    benchmark_version: str = "0"
    prompt_label: str
    sample_turns: str = "all"
    rollouts: int = 1
    fidelity_mean: float = 0.0
    fidelity_std: float = 0.0
    total_steps: int = 0
    seeds: list[SeedResult] = Field(default_factory=list)

    @property
    def seed_values(self) -> list[int]:
        return [s.seed for s in self.seeds]

    @classmethod
    def aggregate(
        cls,
        *,
        benchmark: str,
        benchmark_version: str,
        prompt_label: str,
        sample_turns: str,
        rollouts: int,
        seeds: list[SeedResult],
        run_id: str = "",
        created_at: str = "",
    ) -> BenchRun:
        """Roll per-seed results into a benchmark-level mean + across-seed std.

        The mean is step-weighted (a seed that scored more held-out steps counts proportionally) so
        the overall figure matches what the scorer would report over the pooled steps. The std is
        the population std of the per-seed means, capturing seed-to-seed reproducibility.
        """
        total_steps = sum(s.n_steps for s in seeds)
        if total_steps:
            mean = sum(s.fidelity_mean * s.n_steps for s in seeds) / total_steps
        elif seeds:
            mean = sum(s.fidelity_mean for s in seeds) / len(seeds)
        else:
            mean = 0.0
        return cls(
            run_id=run_id,
            created_at=created_at,
            benchmark=benchmark,
            benchmark_version=benchmark_version,
            prompt_label=prompt_label,
            sample_turns=sample_turns,
            rollouts=rollouts,
            fidelity_mean=mean,
            fidelity_std=pop_std([s.fidelity_mean for s in seeds]),
            total_steps=total_steps,
            seeds=seeds,
        )


def results_dir_for(benchmark_dir: str | Path) -> Path:
    """The results directory for a benchmark (`<benchmark_dir>/results/`)."""
    return Path(benchmark_dir) / RESULTS_DIRNAME


def save_run(run: BenchRun, results_dir: str | Path) -> Path:
    """Write `run` to `<results_dir>/<run_id>.json`, creating the directory if needed."""
    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / f"{run.run_id}.json"
    out.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return out


def load_runs(results_dir: str | Path) -> list[BenchRun]:
    """Load all persisted runs from `results_dir` (empty if the directory doesn't exist)."""
    path = Path(results_dir)
    if not path.exists():
        return []
    return [
        BenchRun.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(path.glob("*.json"))
    ]


__all__ = [
    "BenchRun",
    "SeedResult",
    "RESULTS_DIRNAME",
    "results_dir_for",
    "save_run",
    "load_runs",
]
