"""Tests for the ScoreOnce binding to the eval scorer (fake scorer + provider, no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

import wmh.bench.scoring as scoring
from wmh.bench.definition import JudgeConfig
from wmh.bench.scoring import evaluate_files_once
from wmh.engine.eval import EvalReport


def test_aggregates_rollouts_as_mean_and_std(monkeypatch) -> None:  # noqa: ANN001
    # The scorer returns a different overall fidelity per rollout; the binding should mean+std them.
    fidelities = iter([0.2, 0.8])

    def fake_evaluate(files, prompt, provider, judge, **kwargs):  # noqa: ANN001, ANN003, ANN202
        return EvalReport(overall_fidelity=next(fidelities), total_steps=12)

    monkeypatch.setattr(scoring, "get_provider", lambda config: object())
    monkeypatch.setattr(scoring, "RubricJudge", lambda provider: object())
    monkeypatch.setattr(scoring, "evaluate_files", fake_evaluate)

    score = evaluate_files_once(
        [Path("a.jsonl")],
        "PROMPT",
        JudgeConfig(),
        sample_turns="all",
        rollouts=2,
        temperature=0.7,
        seed=0,
    )
    assert score.rollouts == 2
    assert score.fidelity_mean == 0.5  # mean(0.2, 0.8)
    assert score.fidelity_std == pytest.approx(0.3)  # population std around 0.5
    assert score.n_steps == 12


def test_single_rollout_has_zero_std(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(scoring, "get_provider", lambda config: object())
    monkeypatch.setattr(scoring, "RubricJudge", lambda provider: object())
    monkeypatch.setattr(
        scoring,
        "evaluate_files",
        lambda *a, **k: EvalReport(overall_fidelity=0.9, total_steps=5),
    )
    score = evaluate_files_once(
        [Path("a.jsonl")],
        "PROMPT",
        JudgeConfig(),
        sample_turns="all",
        rollouts=1,
        temperature=0.0,
        seed=0,
    )
    assert score.fidelity_mean == 0.9
    assert score.fidelity_std == 0.0


def test_no_rag_skips_embedder(monkeypatch) -> None:  # noqa: ANN001
    captured: dict[str, object] = {}

    def fake_evaluate(files, prompt, provider, judge, **kwargs):  # noqa: ANN001, ANN003, ANN202
        captured["embedder"] = kwargs.get("embedder")
        return EvalReport(overall_fidelity=0.5, total_steps=3)

    monkeypatch.setattr(scoring, "get_provider", lambda config: object())
    monkeypatch.setattr(scoring, "RubricJudge", lambda provider: object())
    monkeypatch.setattr(scoring, "evaluate_files", fake_evaluate)

    evaluate_files_once(
        [Path("a.jsonl")],
        "PROMPT",
        JudgeConfig(),
        sample_turns="all",
        rollouts=1,
        temperature=0.0,
        seed=0,
        no_rag=True,
    )
    assert captured["embedder"] is None
