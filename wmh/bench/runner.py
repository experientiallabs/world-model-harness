"""Run a benchmark reproducibly and aggregate the rollout distribution as MEAN + STD.

The scoring unit — replay a held-out trace teacher-forced and judge predicted vs. real observation —
belongs to the open-loop eval scorer (`wmh.engine.eval`). This runner sits on top: for each seed in
the benchmark's `EvalConfig`, it scores the prompt once (the scorer draws `rollouts` world-model
samples internally and returns their mean + std), then rolls the per-seed results into one
`SeedResult` apiece. `wmh.bench.results.BenchRun` is the persisted aggregate over all seeds.

`ScoreOnce` is the thin seam over the scorer so this module stays testable with a fake and so the
real `evaluate` API can drop in at one place. See `wmh.bench.scoring` for the production binding.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from wmh.bench.definition import BenchmarkDef, SampleTurns
from wmh.bench.results import BenchRun, SeedResult


class RolloutScore(BaseModel):
    """One seed's fidelity, already aggregated over the seed's `rollouts` world-model samples.

    `fidelity_mean`/`fidelity_std` summarize the rollout distribution (std is 0 when `rollouts == 1`
    or the scorer doesn't yet expose per-rollout scores). `n_steps` is the count of held-out steps
    scored, used to step-weight the benchmark overall.
    """

    fidelity_mean: float = 0.0
    fidelity_std: float = 0.0
    n_steps: int = 0
    rollouts: int = 1


class ScoreOnce(Protocol):
    """Score one prompt against a set of trace files for a single seed.

    The implementation draws `rollouts` world-model samples per scored turn and returns their
    aggregate. `sample_turns` is `"all"` or the per-trace turn cap (Qwen-AgentWorld uses 5).
    """

    def __call__(
        self,
        files: list[Path],
        prompt: str,
        *,
        sample_turns: SampleTurns,
        rollouts: int,
        temperature: float,
        seed: int,
    ) -> RolloutScore: ...


# A hook called after each seed is scored, for progress reporting (seed, index, total).
SeedCallback = Callable[[int, int, int], None]


def run_benchmark(
    bench: BenchmarkDef,
    prompt: str,
    prompt_label: str,
    score_once: ScoreOnce,
    *,
    on_seed: SeedCallback | None = None,
) -> BenchRun:
    """Execute `bench` against `prompt`, scoring once per configured seed, into a `BenchRun`.

    `prompt_label` names the scored prompt in the persisted result (e.g. a model name or prompt
    file). Trace paths are resolved relative to the benchmark dir. The overall fidelity is the
    step-weighted mean of the per-seed means; the across-seed std measures reproducibility.
    """
    files = bench.trace_files()
    cfg = bench.eval
    seeds: list[SeedResult] = []
    for i, seed in enumerate(cfg.seeds):
        score = score_once(
            files,
            prompt,
            sample_turns=cfg.sample_turns,
            rollouts=cfg.rollouts,
            temperature=cfg.temperature,
            seed=seed,
        )
        seeds.append(
            SeedResult(
                seed=seed,
                fidelity_mean=score.fidelity_mean,
                fidelity_std=score.fidelity_std,
                rollouts=score.rollouts,
                n_steps=score.n_steps,
            )
        )
        if on_seed is not None:
            on_seed(seed, i + 1, len(cfg.seeds))

    return BenchRun.aggregate(
        benchmark=bench.name,
        benchmark_version=bench.version,
        prompt_label=prompt_label,
        sample_turns=cfg.sample_turns,
        rollouts=cfg.rollouts,
        seeds=seeds,
    )
