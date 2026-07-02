"""NumericJudge: deterministic scoring for numeric observations.

Semantic judges (`LLMJudge`, `RubricJudge`) grade text equivalence; corpora whose observations are
measurements (robot poses, memory footprints, latencies) need exact numeric comparison instead.
The judge extracts numeric fields from both observations' JSON content, scores each shared field by
relative error, and reports the mean — with per-field scores in `JudgeResult.dimensions` so callers
can also threshold individual fields (e.g. "did it predict the OOM?").
"""

from __future__ import annotations

import json
import math

from pydantic import JsonValue

from wmh.core.types import Observation, Step
from wmh.optimize.judge import JudgeResult


class NumericJudge:
    """Relative-error judge over the numeric fields of JSON observations.

    Each numeric field shared by both observations scores
    `max(0, 1 - |predicted - actual| / (|actual| + eps)) ** 1`, clamped to [0, 1]; booleans and
    error flags must match exactly (score 0 or 1). Fields present in only one observation score 0
    (a missing or hallucinated measurement is wrong, not ignorable). Non-JSON or non-numeric
    content falls back to exact string match so the judge never silently passes garbage.
    """

    def __init__(self, *, tolerance: float = 0.0) -> None:
        # `tolerance` is a relative-error floor under which a field counts as exact (score 1.0),
        # e.g. 0.05 treats predictions within 5% as correct. 0.0 keeps the raw proportional score.
        if tolerance < 0.0:
            raise ValueError(f"tolerance must be >= 0, got {tolerance}")
        self._tolerance = tolerance

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        if predicted.is_error != actual.is_error:
            return JudgeResult(
                score=0.0,
                critique=(
                    f"error-status mismatch: predicted is_error={predicted.is_error}, "
                    f"actual is_error={actual.is_error}"
                ),
            )
        predicted_fields = _numeric_fields(predicted.content)
        actual_fields = _numeric_fields(actual.content)
        if predicted_fields is None or actual_fields is None:
            exact = predicted.content.strip() == actual.content.strip()
            return JudgeResult(
                score=1.0 if exact else 0.0,
                critique="non-numeric content; scored by exact match"
                + ("" if exact else " (contents differ)"),
            )
        if not actual_fields:
            return JudgeResult(score=0.0, critique="actual observation has no numeric fields")

        dimensions: dict[str, float] = {}
        misses: list[str] = []
        for field, actual_value in actual_fields.items():
            if field not in predicted_fields:
                dimensions[field] = 0.0
                misses.append(f"{field}: missing from prediction")
                continue
            dimensions[field] = self._field_score(predicted_fields[field], actual_value)
            if dimensions[field] < 1.0:
                misses.append(
                    f"{field}: predicted {predicted_fields[field]!r} vs actual {actual_value!r}"
                )
        for field in predicted_fields:
            if field not in actual_fields:
                dimensions[field] = 0.0
                misses.append(f"{field}: fabricated (absent from actual)")

        mean = sum(dimensions.values()) / len(dimensions)
        critique = "all numeric fields match" if not misses else "; ".join(misses[:5])
        return JudgeResult(score=mean, critique=critique, dimensions=dimensions)

    def _field_score(self, predicted: float | bool, actual: float | bool) -> float:
        if isinstance(actual, bool) or isinstance(predicted, bool):
            return 1.0 if predicted == actual else 0.0
        if not (math.isfinite(predicted) and math.isfinite(actual)):
            return 1.0 if predicted == actual else 0.0
        relative_error = abs(predicted - actual) / (abs(actual) + 1e-12)
        if relative_error <= self._tolerance:
            return 1.0
        return max(0.0, 1.0 - relative_error)


def _numeric_fields(content: str) -> dict[str, float | bool] | None:
    """Flatten the numeric/boolean leaves of a JSON object into dotted-path fields.

    Returns None when `content` is not a JSON object (caller falls back to exact match).
    """
    try:
        parsed: JsonValue = json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    fields: dict[str, float | bool] = {}
    _collect(parsed, prefix="", into=fields)
    return fields


def _collect(value: JsonValue, *, prefix: str, into: dict[str, float | bool]) -> None:
    if isinstance(value, bool):
        into[prefix] = value
    elif isinstance(value, (int, float)):
        into[prefix] = float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            _collect(child, prefix=f"{prefix}.{key}" if prefix else key, into=into)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _collect(child, prefix=f"{prefix}[{i}]", into=into)
