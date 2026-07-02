"""Robust parsing of model completions into structured values.

Two concerns live here because both the serving engine and the optimizer need them, and `wmh.core`
has no dependencies (so neither imports the other):

- `extract_json_object`: pull the first complete JSON object out of a noisy LLM reply.
- `parse_observation`: turn a world-model completion into a structured `Observation`.

The world-model output contract (see `wmh.core.render.build_env_prompt`) asks the model to reply
with a JSON object ``{"output": str, "is_error": bool, "state_note": str}``. `parse_observation`
is lenient: a reply that is not JSON is treated as a plain-text observation, so a model that ignores
the contract still produces a usable (non-error) observation rather than crashing the step.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, ValidationError

from wmh.core.types import JsonObject, Observation


def extract_json_object(text: str) -> str | None:
    """Return the first complete JSON object substring in `text`, or None if there is none.

    Scans from the first ``{`` to its balanced closing ``}``, tracking string literals and escapes.
    This tolerates ```json fences, surrounding prose, nested objects, and multiple objects (the
    first is returned) — cases a greedy/lazy regex gets wrong.
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


class _RawObservation(BaseModel):
    """Lenient view of the world-model JSON contract before normalization.

    The reasoning-mode fields (`reasoning`, `kb_note`, `ground_query` — see
    `wmh.core.render.output_contract`) default to empty so base-contract replies parse unchanged.
    """

    reasoning: str = ""
    output: str = ""
    is_error: bool = False
    state_note: str = ""
    kb_note: str = ""
    ground_query: str = ""

    def matches_contract(self) -> bool:
        """True when at least one contract field is populated.

        Any JSON object validates against this all-defaults model, so an unrelated object (e.g.
        an API-shaped observation) would otherwise masquerade as an empty contract reply; callers
        use this to fall back to plain-text parsing instead.
        """
        return bool(
            self.output or self.state_note or self.reasoning or self.kb_note or self.ground_query
        )


def parse_observation(text: str) -> Observation:
    """Parse a world-model completion into a structured Observation.

    Prefers the JSON contract ``{"output", "is_error", "state_note"}`` and its reasoning-mode
    superset (``reasoning``/``kb_note``/``ground_query``). ``output`` becomes the observation the
    agent sees; every other populated field is carried in ``metadata`` (``state_note`` feeds the
    session scratchpad, ``kb_note`` the cross-session knowledge base, ``ground_query`` the
    grounder, ``reasoning`` is kept for inspection only). Falls back to treating the whole reply
    as plain observation text when it is not the expected JSON, so an off-contract model still
    yields a usable observation.
    """
    raw = extract_json_object(text)
    if raw is not None:
        try:
            parsed = _RawObservation.model_validate_json(raw)
        except ValidationError:
            parsed = None
        if parsed is not None and parsed.matches_contract():
            metadata: JsonObject = {}
            for key, value in (
                ("state_note", parsed.state_note),
                ("reasoning", parsed.reasoning),
                ("kb_note", parsed.kb_note),
                ("ground_query", parsed.ground_query),
            ):
                if value:
                    metadata[key] = value
            return Observation(content=parsed.output, is_error=parsed.is_error, metadata=metadata)
    return Observation(content=text.strip())


def dumps_observation_contract(observation: Observation) -> str:
    """Render an Observation back into the JSON output contract (used to seed/demo the format)."""
    payload: JsonObject = {"output": observation.content, "is_error": observation.is_error}
    note = observation.metadata.get("state_note")
    if isinstance(note, str) and note:
        payload["state_note"] = note
    return json.dumps(payload)
