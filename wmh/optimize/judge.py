"""The LLM judge that scores a predicted observation against the real one.

The judge is GEPA's fitness signal: it returns a scalar score *and* a natural-language critique,
and the critique is what GEPA reflects on to mutate the prompt.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError

from wmh.core.types import Observation, Step
from wmh.providers.base import Message, Provider

JUDGE_SYSTEM = """You grade a world model that simulates an environment for an AI agent.
Given the agent's action, the ACTUAL observation the real environment returned, and a PREDICTED
observation the world model generated, judge whether the prediction is *functionally equivalent* to
the actual one — i.e. it conveys the same outcome, errors, and salient data the agent would act on.
Ignore cosmetic differences (wording, formatting, ordering, incidental ids). Penalize wrong
outcomes, flipped success/error status, and missing or fabricated salient facts.

Respond with ONLY a JSON object, no prose around it:
{"score": <float 0..1>, "critique": "<one or two sentences: what matched, what diverged, and how \
the prediction should change>"}
Where 1.0 = functionally identical and 0.0 = contradictory or unusable."""


class JudgeResult(BaseModel):
    score: float  # 0..1 semantic match of predicted vs. actual observation
    critique: str  # natural-language feedback; feeds GEPA reflection


@runtime_checkable
class Judge(Protocol):
    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult: ...


class _RawJudgement(BaseModel):
    """Lenient view of the judge's JSON before clamping/normalization."""

    score: float
    critique: str = ""


class LLMJudge:
    """Opus-based semantic-match judge (default fitness signal)."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        user = _build_judge_prompt(predicted, actual, context)
        completion = self._provider.complete(
            JUDGE_SYSTEM,
            [Message(role="user", content=user)],
            temperature=0.0,
            max_tokens=512,
        )
        return _parse_judgement(completion.text)


def _build_judge_prompt(predicted: Observation, actual: Observation, context: Step) -> str:
    action = context.action
    action_desc = action.name or action.content or "(none)"
    return (
        f"AGENT ACTION ({action.kind.value}): {action_desc}\n"
        f"ACTION ARGUMENTS: {json.dumps(action.arguments, sort_keys=True, default=str)}\n\n"
        f"ACTUAL OBSERVATION (is_error={actual.is_error}):\n{actual.content}\n\n"
        f"PREDICTED OBSERVATION (is_error={predicted.is_error}):\n{predicted.content}\n"
    )


def _parse_judgement(text: str) -> JudgeResult:
    """Robustly parse the judge's reply into a JudgeResult.

    Accepts a bare JSON object, JSON inside a ```json fence, or JSON embedded in surrounding prose.
    Falls back to a neutral-but-flagged failure rather than raising, so a single malformed reply
    does not abort a whole GEPA run.
    """
    raw = _extract_json(text)
    if raw is not None:
        try:
            parsed = _RawJudgement.model_validate_json(raw)
            return JudgeResult(score=_clamp(parsed.score), critique=parsed.critique.strip())
        except ValidationError:
            pass
    return JudgeResult(
        score=0.0,
        critique=f"Unparseable judge response; treated as failure. Raw: {text.strip()[:200]}",
    )


def _extract_json(text: str) -> str | None:
    """Pull the first complete JSON object out of a model reply.

    Scans for the first ``{`` and returns up to its balanced closing ``}`` (tracking string
    literals and escapes). This tolerates ```json fences, surrounding prose, nested objects, and
    multiple objects (we take the first) — cases a greedy/lazy regex gets wrong.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))
