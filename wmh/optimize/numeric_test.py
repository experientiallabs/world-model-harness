"""Tests for NumericJudge."""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from wmh.core.types import Action, ActionKind, Observation, Step
from wmh.optimize.judge import Judge
from wmh.optimize.numeric import NumericJudge

STEP = Step(
    action=Action(kind=ActionKind.TOOL_CALL, name="measure", arguments={}),
    observation=Observation(content="{}"),
)


def _obs(payload: Mapping[str, object], *, is_error: bool = False) -> Observation:
    return Observation(content=json.dumps(payload), is_error=is_error)


def test_satisfies_judge_protocol() -> None:
    assert isinstance(NumericJudge(), Judge)


def test_exact_match_scores_one() -> None:
    result = NumericJudge().score(
        _obs({"peak_mem_gb": 12.5, "oom": False}), _obs({"peak_mem_gb": 12.5, "oom": False}), STEP
    )
    assert result.score == 1.0
    assert result.dimensions == {"peak_mem_gb": 1.0, "oom": 1.0}


def test_relative_error_scales_score() -> None:
    result = NumericJudge().score(_obs({"latency_s": 11.0}), _obs({"latency_s": 10.0}), STEP)
    assert result.dimensions["latency_s"] == pytest.approx(0.9, abs=1e-6)


def test_tolerance_forgives_small_errors() -> None:
    judge = NumericJudge(tolerance=0.05)
    result = judge.score(_obs({"latency_s": 10.3}), _obs({"latency_s": 10.0}), STEP)
    assert result.score == 1.0


def test_boolean_mismatch_is_zero_even_when_numbers_match() -> None:
    result = NumericJudge().score(
        _obs({"peak_mem_gb": 12.5, "oom": True}), _obs({"peak_mem_gb": 12.5, "oom": False}), STEP
    )
    assert result.dimensions["oom"] == 0.0
    assert result.score == pytest.approx(0.5)
    assert "oom" in result.critique


def test_missing_and_fabricated_fields_score_zero() -> None:
    result = NumericJudge().score(_obs({"made_up": 1.0}), _obs({"peak_mem_gb": 12.5}), STEP)
    assert result.dimensions == {"peak_mem_gb": 0.0, "made_up": 0.0}
    assert result.score == 0.0
    assert "missing" in result.critique and "fabricated" in result.critique


def test_error_status_mismatch_is_zero() -> None:
    result = NumericJudge().score(
        _obs({"x": 1.0}, is_error=True), _obs({"x": 1.0}, is_error=False), STEP
    )
    assert result.score == 0.0
    assert "error-status" in result.critique


def test_nested_and_list_fields_flatten() -> None:
    payload = {"gpu": {"mem_gb": 40.0}, "per_layer": [1.0, 2.0]}
    result = NumericJudge().score(_obs(payload), _obs(payload), STEP)
    assert set(result.dimensions) == {"gpu.mem_gb", "per_layer[0]", "per_layer[1]"}
    assert result.score == 1.0


def test_non_json_falls_back_to_exact_match() -> None:
    same = NumericJudge().score(
        Observation(content="42 files"), Observation(content="42 files"), STEP
    )
    diff = NumericJudge().score(
        Observation(content="42 files"), Observation(content="41 files"), STEP
    )
    assert same.score == 1.0
    assert diff.score == 0.0
    assert "exact match" in diff.critique


def test_negative_tolerance_rejected() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        NumericJudge(tolerance=-0.1)
