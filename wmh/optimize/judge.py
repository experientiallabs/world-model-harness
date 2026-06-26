"""The LLM judge that scores a predicted observation against the real one.

The judge is GEPA's fitness signal: it returns a scalar score *and* a natural-language critique,
and the critique is what GEPA reflects on to mutate the prompt.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.core.types import Observation, Step
from wmh.providers.base import Message, Provider

# A stable substring of JUDGE_SYSTEM used to recognize judge calls when attributing run cost
# (wmh.tracking.metered). Kept as a named constant so the prompt and the classifier never drift:
# JUDGE_SYSTEM must always contain JUDGE_MARKER (pinned by judge_test.py).
JUDGE_MARKER = "grade a world model"

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
    # Per-dimension scores (0..1), populated by RubricJudge; empty for the plain LLMJudge. `score`
    # stays the single headline number either way, so callers that read only `score` are unaffected.
    dimensions: dict[str, float] = Field(default_factory=dict)


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
    raw = extract_json_object(text)
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


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


# --- RubricJudge: the open-loop fidelity scorer (Qwen-AgentWorld-style) ---------------------------

# The five fidelity dimensions, scored independently in 0..1. Modeled on Qwen-AgentWorld
# (arXiv 2606.24597) "AgentWorldBench" rubric; the headline `score` is their mean.
RUBRIC_DIMENSIONS = ("format", "factuality", "consistency", "realism", "quality")

RUBRIC_JUDGE_MARKER = "grade a world model"  # shares the cost-attribution marker (see [[judge]])

RUBRIC_JUDGE_SYSTEM = """You grade a world model that simulates an environment for an AI agent. You
see the agent's action, the ACTUAL observation the real environment returned, and a PREDICTED
observation the world model generated. Score the prediction's fidelity to the actual observation on
five independent dimensions, each from 0.0 to 1.0:

- format: same shape/structure/encoding the environment uses (JSON shape, field names, exit status).
- factuality: conveys the same outcome, errors, and SALIENT data the agent would act on.
- consistency: coherent with the action and the environment's established behavior.
- realism: looks like a real response this environment would emit (not an explanation or apology).
- quality: overall, how usable this prediction is as a stand-in for the real observation.

CONTENT TYPE matters — infer it from the action and observation:
- DETERMINISTIC / computed content (a file's contents via `cat`, a command's computed stdout, a
  lookup of state that exists) MUST match the actual values to score high on factuality — wrong
  computed values are unambiguously wrong even if well-formatted.
- VOLATILE / incidental content (PIDs, timestamps, random ids, ordering of unordered output) should
  be judged on plausibility and format only — a different-but-plausible value is fine.

Respond with ONLY a JSON object, no prose:
{"format": <0..1>, "factuality": <0..1>, "consistency": <0..1>, "realism": <0..1>,
 "quality": <0..1>, "critique": "<one or two sentences: what matched, what diverged>"}"""


class _RawRubric(BaseModel):
    """Lenient view of the rubric judge's JSON; missing dims default to 0.0."""

    format: float = 0.0
    factuality: float = 0.0
    consistency: float = 0.0
    realism: float = 0.0
    quality: float = 0.0
    critique: str = ""


class RubricJudge:
    """Reference-grounded 5-dimension fidelity judge for open-loop evaluation.

    Unlike `LLMJudge` (a single functional-equivalence score used as GEPA's fitness signal), this
    scores the five `RUBRIC_DIMENSIONS` and reports their mean as the headline `score`, with the
    per-dimension breakdown in `JudgeResult.dimensions`. The prompt instructs the deterministic-vs-
    volatile content split so computed outputs are held to exact correctness while incidental values
    (PIDs, timestamps) are judged on plausibility.
    """

    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    def score(self, predicted: Observation, actual: Observation, context: Step) -> JudgeResult:
        user = _build_judge_prompt(predicted, actual, context)
        completion = self._provider.complete(
            RUBRIC_JUDGE_SYSTEM,
            [Message(role="user", content=user)],
            temperature=0.0,
            max_tokens=512,
        )
        return _parse_rubric(completion.text)


def _parse_rubric(text: str) -> JudgeResult:
    """Parse the rubric judge's reply into a JudgeResult (headline score = mean of dimensions)."""
    raw = extract_json_object(text)
    if raw is not None:
        try:
            parsed = _RawRubric.model_validate_json(raw)
        except ValidationError:
            parsed = None
        if parsed is not None:
            dims = {d: _clamp(getattr(parsed, d)) for d in RUBRIC_DIMENSIONS}
            mean = sum(dims.values()) / len(dims)
            return JudgeResult(score=mean, critique=parsed.critique.strip(), dimensions=dims)
    return JudgeResult(
        score=0.0,
        critique=f"Unparseable rubric response; treated as failure. Raw: {text.strip()[:200]}",
    )
