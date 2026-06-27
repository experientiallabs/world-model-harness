"""Production binding of the `ScoreOnce` seam to the open-loop eval scorer.

This is the ONE place the benchmark layer touches the scorer API. It builds the reference-grounded
`RubricJudge` (the open-loop fidelity grader — five dimensions, mean as the headline score) and
replays the held-out steps via `wmh.engine.eval.evaluate_files`, which returns an `EvalReport` whose
`overall_fidelity`/`overall_std`/`total_steps` we read directly.

`rollouts`/`temperature` are reproducibility knobs reserved for when the providers forward a
sampling temperature: `evaluate_files` is deterministic today (no temperature plumbing), so extra
rollouts reproduce identical scores. We still loop `rollouts` times so the mean±std machinery is
wired end-to-end — at the committed `rollouts=1` that is a single call, and the per-seed std is 0
until either `rollouts` > 1 with temperature support, or multiple `seeds` produce a spread.
"""

from __future__ import annotations

from pathlib import Path

from wmh.bench._stats import pop_std
from wmh.bench.definition import JudgeConfig, SampleTurns
from wmh.bench.runner import RolloutScore
from wmh.engine.eval import evaluate_files
from wmh.optimize.judge import RubricJudge
from wmh.providers import ProviderConfig, ProviderKind, get_provider
from wmh.retrieval import HashingEmbedder


def evaluate_files_once(
    files: list[Path],
    prompt: str,
    judge_config: JudgeConfig,
    *,
    sample_turns: SampleTurns,
    rollouts: int,
    temperature: float,
    seed: int,
    train_split: float = 0.7,
    top_k: int = 5,
    no_rag: bool = False,
    embed_dim: int = 512,
) -> RolloutScore:
    """Score `prompt` against `files` for one seed, aggregating `rollouts` samples as mean + std.

    Builds the serve provider + `RubricJudge` from `judge_config` (the benchmark pins the grader
    for reproducibility), then replays the held-out steps once per rollout, forwarding
    `sample_turns` and `seed` to the scorer. `temperature` is accepted for forward-compatibility but
    inert until the providers forward a sampling temperature (see the module docstring), so rollouts
    currently reproduce identical scores.
    """
    provider = get_provider(
        ProviderConfig(
            kind=ProviderKind(judge_config.provider),
            model=judge_config.model,
            region=judge_config.region,
        )
    )
    judge = RubricJudge(provider)
    embedder = None if no_rag else HashingEmbedder(dim=embed_dim)

    fidelities: list[float] = []
    total_steps = 0
    for _ in range(max(1, rollouts)):
        report = evaluate_files(
            files,
            prompt,
            provider,
            judge,
            embedder=embedder,
            train_split=train_split,
            top_k=top_k,
            sample_turns=sample_turns,
            seed=seed,
        )
        fidelities.append(report.overall_fidelity)
        total_steps = report.total_steps  # stable across rollouts (same held-out split)

    _ = temperature  # reserved; inert until the providers forward a sampling temperature
    return RolloutScore(
        fidelity_mean=sum(fidelities) / len(fidelities),
        fidelity_std=pop_std(fidelities),
        n_steps=total_steps,
        rollouts=len(fidelities),
    )


__all__ = ["evaluate_files_once"]
