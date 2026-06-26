"""Production binding of the `ScoreOnce` seam to the open-loop eval scorer.

This is the ONE place that touches the scorer API, so when the eval-scorer chat's open-loop
`evaluate_files(..., rollouts, temperature, sample_turns, seed)` lands we collapse this to a single
call here and nowhere else. On `main` today `evaluate_files` doesn't yet take those knobs, so we
draw `rollouts` independent samples ourselves and aggregate their overall fidelity as mean + std —
exactly the rollout distribution the benchmark reports. `sample_turns`/`seed` are part of the
reproducible contract and flow through unchanged until that scorer merges.

    # TODO: replace with the open-loop wmh.engine.eval.evaluate_files once chat 1 (eval scorer)
    # merges. Its EvalReport already exposes the rollout aggregate, so this whole loop collapses to:
    #   report = evaluate_files(files, prompt, provider, judge, embedder=embedder,
    #                           train_split=..., top_k=..., rollouts=rollouts,
    #                           temperature=temperature, sample_turns=sample_turns, seed=seed)
    #   return RolloutScore(fidelity_mean=report.overall_fidelity,
    #                       fidelity_std=report.overall_std, n_steps=report.total_steps,
    #                       rollouts=report.rollouts)
"""

from __future__ import annotations

from pathlib import Path

from wmh.bench._stats import pop_std
from wmh.bench.definition import JudgeConfig, SampleTurns
from wmh.bench.runner import RolloutScore
from wmh.engine.eval import evaluate_files
from wmh.optimize.judge import LLMJudge
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

    Builds the judge/serve provider from `judge_config` (the benchmark pins the grader for
    reproducibility), then replays the held-out steps once per rollout. At `temperature` > 0 each
    rollout's overall fidelity differs, so we report the mean and population std over them.
    `sample_turns`/`seed` ride along for the open-loop scorer (see the module TODO).
    """
    provider = get_provider(
        ProviderConfig(
            kind=ProviderKind(judge_config.provider),
            model=judge_config.model,
            region=judge_config.region,
        )
    )
    judge = LLMJudge(provider)
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
        )
        fidelities.append(report.overall_fidelity)
        total_steps = report.total_steps  # stable across rollouts (same held-out split)

    _ = (sample_turns, temperature, seed)  # forwarded to the open-loop scorer on merge (see TODO)
    return RolloutScore(
        fidelity_mean=sum(fidelities) / len(fidelities),
        fidelity_std=pop_std(fidelities),
        n_steps=total_steps,
        rollouts=len(fidelities),
    )


__all__ = ["evaluate_files_once"]
