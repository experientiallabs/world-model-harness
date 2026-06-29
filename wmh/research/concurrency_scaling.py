"""Concurrency scaling law: batch wall-clock vs. how many scenarios run at once.

The world model and the real sandbox both reconstruct the SAME held-out scenarios; the question
this experiment answers is **how the wall-clock of each side changes as we raise concurrency W**,
and therefore the *time differential* T_real(W) / T_world(W) at each level. The world-model side
parallelizes near-perfectly (open-loop steps are independent, each scenario is its own provider +
tracker — see `wmh.bench.run_scenario`); the real side is bounded by container/process standup. The
crossover and saturation points are the headline.

This reuses the deployed primitives rather than reimplementing them: the caller passes a
`world_runner(level) -> WorldBatch` and (optionally) a `real_runner(level) -> RealBatch` that each
run a fixed batch of N scenarios at concurrency `level` and report the batch wall-clock. The runner
script (`scripts/run_concurrency_scaling.py`) wires those to `run_scenario` /
`wmh.bench.side_by_side.run_real_sandbox` over a `ThreadPoolExecutor`; the unit tests pass fakes, so
no network/Docker is assumed here. Aggregation (mean ± std across trials) reuses
`wmh.research.ablation._mean_std`, so the error bars match the rest of the research harness.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum

from pydantic import BaseModel, Field

from wmh.research.ablation import _mean_std


class Side(StrEnum):
    """Which half/halves of the comparison to time at each concurrency level."""

    BOTH = "both"  # time the world-model batch AND the real-sandbox batch (the differential)
    WORLD = "world"  # world-model batch only (cheap; no sandbox setup)
    REAL = "real"  # real-sandbox batch only


class WorldBatch(BaseModel):
    """One world-model batch run: N scenarios at one concurrency level, timed + metered."""

    wall_seconds: float  # batch wall-clock (the headline — overlaps the N scenarios)
    work_seconds: float = 0.0  # summed per-scenario model time (would-be sequential cost)
    ok: int = 0  # scenarios that completed
    total: int = 0  # scenarios attempted
    tokens: int = 0
    cost_usd: float = 0.0
    fidelity: float = 0.0  # step-weighted error-flag agreement across the batch (0..1)


class RealBatch(BaseModel):
    """One real-sandbox batch run: N sandboxes at one concurrency level, timed."""

    wall_seconds: float  # batch wall-clock (overlaps the N sandboxes)
    work_seconds: float = 0.0  # summed per-sandbox wall (would-be sequential cost)
    ok: int = 0
    total: int = 0


class ConcurrencyTrial(BaseModel):
    """One (level, trial) measurement: the batch(es) timed at concurrency `level`."""

    level: int
    trial: int
    world: WorldBatch | None = None
    real: RealBatch | None = None


class ConcurrencyPoint(BaseModel):
    """Aggregated outcome at one concurrency level: mean ± std across trials, plus the derived law.

    `speedup`/`efficiency` describe the world-model side (the parallelizable one) relative to the
    level-1 baseline; `differential` is T_real / T_world at this level (>1 means the real sandbox is
    slower — the world model's advantage). Derived fields are 0.0 when the inputs to compute them
    are missing (e.g. no level-1 baseline, or a side wasn't timed).
    """

    level: int
    trials: int = 0
    world_wall_mean: float = 0.0
    world_wall_std: float = 0.0
    real_wall_mean: float = 0.0
    real_wall_std: float = 0.0
    world_fidelity_mean: float = 0.0
    world_tokens_mean: float = 0.0
    world_cost_mean: float = 0.0
    speedup: float = 0.0  # T_world(1) / T_world(level)
    efficiency: float = 0.0  # speedup / level (1.0 = perfect scaling)
    differential: float = 0.0  # T_real(level) / T_world(level)

    def summary(self) -> str:
        parts = [f"W={self.level:<3}"]
        if self.world_wall_mean:
            parts.append(f"world={self.world_wall_mean:.1f}±{self.world_wall_std:.1f}s")
            parts.append(f"speedup={self.speedup:.2f}x eff={self.efficiency:.0%}")
        if self.real_wall_mean:
            parts.append(f"real={self.real_wall_mean:.1f}±{self.real_wall_std:.1f}s")
        if self.world_wall_mean and self.real_wall_mean:
            parts.append(f"diff={self.differential:.2f}x")
        return "  ".join(parts)


class ConcurrencyScalingReport(BaseModel):
    """The full experiment: one `ConcurrencyPoint` per concurrency level. The canonical artifact."""

    name: str = "concurrency-scaling"
    benchmark: str = ""
    side: Side = Side.BOTH
    scenarios: int = 0  # batch size N held fixed across levels
    levels: list[int] = Field(default_factory=list)
    points: list[ConcurrencyPoint] = Field(default_factory=list)

    def best_speedup(self) -> ConcurrencyPoint | None:
        """The level with the highest world-model speedup (None if empty)."""
        return max(self.points, key=lambda p: p.speedup, default=None)


# Per-level runners: given a concurrency level, run the fixed N-scenario batch and report timing.
WorldRunner = Callable[[int], WorldBatch]
RealRunner = Callable[[int], RealBatch]
# Progress hook called after each level aggregates: (point) -> None.
PointCallback = Callable[[ConcurrencyPoint], None]


def _aggregate_level(
    level: int, trials: list[ConcurrencyTrial], baseline_world_wall: float | None
) -> ConcurrencyPoint:
    """Roll up a level's trials into a `ConcurrencyPoint` (mean ± std + the derived scaling law)."""
    world = [t.world for t in trials if t.world is not None]
    real = [t.real for t in trials if t.real is not None]
    world_wall_mean, world_wall_std = _mean_std([w.wall_seconds for w in world])
    real_wall_mean, real_wall_std = _mean_std([r.wall_seconds for r in real])
    fidelity_mean, _ = _mean_std([w.fidelity for w in world])
    tokens_mean, _ = _mean_std([float(w.tokens) for w in world])
    cost_mean, _ = _mean_std([w.cost_usd for w in world])

    speedup = (
        baseline_world_wall / world_wall_mean
        if baseline_world_wall and world_wall_mean
        else 0.0
    )
    efficiency = speedup / level if speedup and level else 0.0
    differential = real_wall_mean / world_wall_mean if world_wall_mean and real_wall_mean else 0.0
    return ConcurrencyPoint(
        level=level,
        trials=len(trials),
        world_wall_mean=world_wall_mean,
        world_wall_std=world_wall_std,
        real_wall_mean=real_wall_mean,
        real_wall_std=real_wall_std,
        world_fidelity_mean=fidelity_mean,
        world_tokens_mean=tokens_mean,
        world_cost_mean=cost_mean,
        speedup=speedup,
        efficiency=efficiency,
        differential=differential,
    )


def run_concurrency_scaling(
    world_runner: WorldRunner,
    real_runner: RealRunner | None,
    *,
    levels: Sequence[int],
    scenarios: int,
    trials: int = 1,
    side: Side = Side.BOTH,
    on_point: PointCallback | None = None,
) -> ConcurrencyScalingReport:
    """Sweep concurrency `levels`, timing the fixed N-scenario batch `trials` times at each.

    For each level we run the world-model batch (unless `side` is REAL) and the real-sandbox batch
    (when `side` is BOTH/REAL and a `real_runner` is given), repeated `trials` times for error bars,
    then aggregate to a `ConcurrencyPoint`. The world-model speedup is measured against the FIRST
    level's mean world wall-clock, so put the baseline (usually 1) first in `levels`. `on_point` is
    called after each level for live progress.

    Levels run in the given order, trials in sequence; a run is reproducible given the same runners
    and levels. The runners own the actual concurrency (a `ThreadPoolExecutor` of width `level`);
    the world side is safe to parallelize: each scenario uses its own provider + `RunTracker`.
    """
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if scenarios < 1:
        raise ValueError("scenarios must be at least 1")
    want_world = side in (Side.BOTH, Side.WORLD)
    want_real = side in (Side.BOTH, Side.REAL) and real_runner is not None

    points: list[ConcurrencyPoint] = []
    baseline_world_wall: float | None = None
    for level in levels:
        if level < 1:
            raise ValueError(f"concurrency level must be at least 1, got {level}")
        level_trials: list[ConcurrencyTrial] = []
        for trial in range(trials):
            world = world_runner(level) if want_world else None
            real = real_runner(level) if want_real and real_runner is not None else None
            level_trials.append(
                ConcurrencyTrial(level=level, trial=trial, world=world, real=real)
            )
        point = _aggregate_level(level, level_trials, baseline_world_wall)
        if baseline_world_wall is None and point.world_wall_mean:
            # First level with a world measurement is the baseline; backfill its own speedup.
            baseline_world_wall = point.world_wall_mean
            point = _aggregate_level(level, level_trials, baseline_world_wall)
        points.append(point)
        if on_point is not None:
            on_point(point)
    return ConcurrencyScalingReport(
        side=side, scenarios=scenarios, levels=list(levels), points=points
    )


__all__ = [
    "ConcurrencyPoint",
    "ConcurrencyScalingReport",
    "ConcurrencyTrial",
    "RealBatch",
    "RealRunner",
    "Side",
    "WorldBatch",
    "WorldRunner",
    "run_concurrency_scaling",
]
