"""Tests for the SIB->OTel transcript converter.

Run explicitly (the project's default testpaths is `wmh/`): `uv run pytest scripts/ -q`.
"""

from __future__ import annotations

from scripts.sib_to_otel import _parse_env_reply, transcript_to_spans
from wmh.core.types import JsonObject
from wmh.ingest import get_adapter


def _attr(span: JsonObject, key: str) -> str | None:
    attributes = span["attributes"]
    assert isinstance(attributes, list)
    for a in attributes:
        if isinstance(a, dict) and a.get("key") == key:
            value = a.get("value")
            assert isinstance(value, dict)
            string_value = value.get("stringValue")
            return string_value if isinstance(string_value, str) else None
    return None


def test_parse_env_reply_flags_nonzero_returncode() -> None:
    assert _parse_env_reply("<returncode>0</returncode><output>ok</output>") == ("ok", False)
    out, err = _parse_env_reply("<returncode>1</returncode><output>boom</output>")
    assert out == "boom" and err is True


def test_transcript_maps_turns_to_llm_and_tool_spans() -> None:
    transcript = {
        "messages": [
            {"role": "system", "content": "you are an agent"},
            {"role": "user", "content": "Customer request: book r_900 for u_kath"},
            {"role": "assistant", "content": "ok\n```sib_bash\nget_user u_kath\n```"},
            {
                "role": "user",
                "content": '<returncode>0</returncode><output>{"ok": true}</output>',
            },
            {"role": "assistant", "content": "done\n```sib_bash\necho SIB_SUBMIT\n```"},
        ]
    }
    spans = transcript_to_spans(transcript, trace_id="a" * 32)
    # 2 assistant turns -> 2 LLM spans; only the first has a following env reply -> 1 tool span.
    kinds = [_attr(s, "gen_ai.operation.name") for s in spans]
    assert kinds == ["chat", "execute_tool", "chat"]
    assert _attr(spans[0], "gen_ai.tool.call.arguments") == '{"command": "get_user u_kath"}'
    prompt = _attr(spans[0], "gen_ai.prompt")
    assert prompt is not None and prompt.startswith("Customer request")
    assert _attr(spans[1], "gen_ai.tool.message") == '{"ok": true}'


def test_roundtrips_through_the_otel_adapter(tmp_path) -> None:  # noqa: ANN001 - pytest fixture
    transcript = {
        "messages": [
            {"role": "user", "content": "Customer request: status of r_042"},
            {"role": "assistant", "content": "checking\n```sib_bash\nget_reservation r_042\n```"},
            {"role": "user", "content": "<returncode>0</returncode><output>confirmed</output>"},
        ]
    }
    import json

    spans = transcript_to_spans(transcript, trace_id="b" * 32)
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(s) for s in spans), encoding="utf-8")

    traces = get_adapter("otel-genai").from_file(str(path))
    assert len(traces) == 1
    step = traces[0].steps[0]
    assert step.action.name == "bash"
    assert step.action.arguments == {"command": "get_reservation r_042"}
    assert step.observation.content == "confirmed"
    assert step.task is not None and step.task.startswith("Customer request")
