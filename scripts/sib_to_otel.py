"""Convert self-improvement-bench (SIB) message-traces into OTel GenAI span JSONL.

The point is to reuse trace data we already have, not to couple to SIB: this transforms SIB's
saved agent transcripts (`results/.../traces/<task>.json`) into the bare-span OTLP-JSON-lines shape
that `wmh.ingest.otel_genai` already ingests. One transcript -> one trace; each agent bash turn ->
an LLM span (a `bash` tool call), each following environment reply -> an `execute_tool` span.

Usage:
    python scripts/sib_to_otel.py <sib_traces_dir> <out.jsonl>

where <sib_traces_dir> holds SIB transcript files like:
    {"messages": [{"role": "system"|"user"|"assistant", "content": "..."}], "exit_status": ...}

The agent answers with one ```sib_bash``` fenced command per assistant turn; the environment's reply
is a `user` message wrapping `<returncode>N</returncode>` and `<output>...</output>`.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

from pydantic import JsonValue

from wmh.core.types import JsonObject

# The agent emits exactly one fenced command per turn; capture its body.
_BASH_BLOCK = re.compile(r"```sib_bash\s*(.*?)```", re.DOTALL)
_RETURNCODE = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>", re.DOTALL)
_OUTPUT = re.compile(r"<output>(.*?)</output>", re.DOTALL)


class _Message(TypedDict):
    role: str
    content: str


def _string_attr(key: str, value: str) -> JsonObject:
    return {"key": key, "value": {"stringValue": value}}


def _extract_command(assistant_content: str) -> str | None:
    """Pull the bash command out of an assistant turn, or None if it has no fenced block."""
    match = _BASH_BLOCK.search(assistant_content)
    if match is None:
        return None
    return match.group(1).strip()


def _parse_env_reply(user_content: str) -> tuple[str, bool]:
    """Return (output_text, is_error) for an environment reply. Non-zero returncode -> error."""
    rc_match = _RETURNCODE.search(user_content)
    is_error = rc_match is not None and int(rc_match.group(1)) != 0
    out_match = _OUTPUT.search(user_content)
    output = out_match.group(1).strip() if out_match else user_content.strip()
    return output, is_error


def _span(
    *,
    trace_id: str,
    span_id: str,
    name: str,
    start_nano: int,
    attributes: list[JsonObject],
    is_error: bool = False,
) -> JsonObject:
    """Build one bare OTLP-JSON span the otel-genai adapter understands."""
    status: JsonObject = {"code": "STATUS_CODE_ERROR" if is_error else "STATUS_CODE_OK"}
    span: JsonObject = {
        "traceId": trace_id,
        "spanId": span_id,
        "parentSpanId": "",
        "name": name,
        "startTimeUnixNano": start_nano,
        "endTimeUnixNano": start_nano + 1,
        "status": status,
        "attributes": list(attributes),
    }
    return span


def transcript_to_spans(transcript: Mapping[str, JsonValue], trace_id: str) -> list[JsonObject]:
    """Turn one SIB transcript into ordered OTel GenAI spans.

    Mapping (DreamGym (state, action) -> observation):
      - first user message            -> the task (gen_ai.prompt on the first LLM span)
      - assistant ```sib_bash``` turn -> LLM span: a `bash` tool call (arguments={"command": ...})
      - following env `user` reply    -> execute_tool span: output + error status
    """
    raw_messages = transcript.get("messages")
    if not isinstance(raw_messages, list):
        return []
    messages: list[_Message] = [
        {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
        for m in raw_messages
        if isinstance(m, dict)
    ]

    # The task is the first user request (the customer ask), before any tool output.
    task = next(
        (
            m["content"]
            for m in messages
            if m["role"] == "user" and "<returncode>" not in m["content"]
        ),
        None,
    )

    spans: list[JsonObject] = []
    clock = 0
    seq = 0
    first_llm = True
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg["role"] != "assistant":
            i += 1
            continue
        command = _extract_command(msg["content"])
        if command is None:
            i += 1
            continue

        llm_attrs = [
            _string_attr("gen_ai.operation.name", "chat"),
            _string_attr("gen_ai.request.model", "sib-agent"),
            _string_attr("gen_ai.tool.name", "bash"),
            _string_attr("gen_ai.tool.call.arguments", json.dumps({"command": command})),
        ]
        if first_llm and task is not None:
            llm_attrs.append(_string_attr("gen_ai.prompt", task))
            first_llm = False
        spans.append(
            _span(
                trace_id=trace_id,
                span_id=f"{trace_id[:8]}{seq:04x}",
                name="chat bash",
                start_nano=clock,
                attributes=llm_attrs,
            )
        )
        clock += 10
        seq += 1

        # Pair with the next environment reply, if present (the final SUBMIT turn has none).
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        if nxt is not None and nxt["role"] == "user":
            output, is_error = _parse_env_reply(nxt["content"])
            spans.append(
                _span(
                    trace_id=trace_id,
                    span_id=f"{trace_id[:8]}{seq:04x}",
                    name="execute_tool bash",
                    start_nano=clock,
                    attributes=[
                        _string_attr("gen_ai.operation.name", "execute_tool"),
                        _string_attr("gen_ai.tool.name", "bash"),
                        _string_attr("gen_ai.tool.message", output),
                    ],
                    is_error=is_error,
                )
            )
            clock += 10
            seq += 1
            i += 2
        else:
            i += 1
    return spans


def _trace_id_for(path: Path, index: int) -> str:
    """Deterministic 32-hex trace id derived from the file stem (stable across runs)."""
    stem = path.stem
    digest = "".join(f"{ord(c):02x}" for c in stem)[:24]
    return f"{digest:0<24}{index:08x}"[:32]


def convert_dir(traces_dir: Path) -> list[JsonObject]:
    """Convert every transcript in `traces_dir` into a flat list of spans across all traces."""
    all_spans: list[JsonObject] = []
    for index, path in enumerate(sorted(traces_dir.glob("*.json"))):
        transcript = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(transcript, dict):
            continue
        trace_id = _trace_id_for(path, index)
        all_spans.extend(transcript_to_spans(transcript, trace_id))
    return all_spans


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    traces_dir = Path(argv[1])
    out_path = Path(argv[2])
    spans = convert_dir(traces_dir)
    with out_path.open("w", encoding="utf-8") as fh:
        for span in spans:
            fh.write(json.dumps(span) + "\n")
    print(f"wrote {len(spans)} spans from {traces_dir} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
