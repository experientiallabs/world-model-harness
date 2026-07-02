"""Tests for corpus hygiene: detecting trajectories that escaped the task workspace."""

from __future__ import annotations

import json
from pathlib import Path

from environment_capture.hygiene import (
    host_escape_findings,
    partition_contained,
    scan_spans_jsonl,
)
from environment_capture.trajectory import StepRecord, Task, ToolCall, Trajectory


def _trajectory(command: str, output: str) -> Trajectory:
    return Trajectory(
        task=Task(task_id="t0", prompt="q", data={}),
        steps=[
            StepRecord(
                action=ToolCall(name="bash", arguments={"command": command}),
                output=output,
                is_error=False,
            )
        ],
    )


def test_workspace_contained_trajectory_is_clean() -> None:
    clean = _trajectory(
        "ls docs && grep -RinE 'Net sales|/shares/' docs/*.txt | head -5",
        "docs/a.txt:12: Net sales were $1,577",
    )
    assert host_escape_findings(clean) == []


def test_host_targeting_commands_are_flagged() -> None:
    for command in (
        "ls -R /home 2>/dev/null | head -50",
        "find / -name database.db 2>/dev/null",
        "ls ~",
        "ls -la $HOME",
        "cd .. && ls",
        "cat /Users/someone/.ssh/config",
        "cd /root",
    ):
        findings = host_escape_findings(_trajectory(command, "whatever"))
        assert findings, f"expected flag for command: {command}"
        assert findings[0].field == "command"


def test_host_content_in_observations_is_flagged() -> None:
    for output in (
        "drwx------@ 3 someuser staff 96 Jul 1 22:06 Desktop\n.ssh\nid_ecdsa.pub",
        'File "/Users/someone/anaconda3/lib/python3.11/re/__init__.py", line 176',
        "bash: line 0: cd: /root: No such file or directory",
        "/home/user/project/node_modules",
    ):
        findings = host_escape_findings(_trajectory("echo hi", output))
        assert findings, f"expected flag for output: {output[:40]}"
        assert findings[0].field == "output"


def test_machine_username_in_observation_is_flagged() -> None:
    """`ls -l` ownership columns leak the machine username even without leaving the workspace;
    the detector learns the CURRENT user/home at runtime so no personal string lives in code."""
    import getpass

    user = getpass.getuser()
    listing = f"total 0\ndrwx------@ 3 {user}  staff  96 Jul  1 22:06 ."
    findings = host_escape_findings(_trajectory("ls -la", listing))
    assert findings and findings[0].field == "output"
    assert findings[0].marker == user


def test_partition_contained_splits_and_preserves_order() -> None:
    clean_1 = _trajectory("ls docs", "a.txt")
    dirty = _trajectory("ls ~", "Desktop")
    clean_2 = _trajectory("cat docs/a.txt", "text")
    clean, flagged = partition_contained([clean_1, dirty, clean_2])
    assert clean == [clean_1, clean_2]
    assert flagged == [dirty]


def test_scan_spans_jsonl_maps_trace_ids_to_findings(tmp_path: Path) -> None:
    def span(trace_id: str, key: str, value: str) -> dict[str, object]:
        return {
            "traceId": trace_id,
            "spanId": f"{trace_id}-s",
            "attributes": [{"key": key, "value": {"stringValue": value}}],
        }

    path = tmp_path / "traces.otel.jsonl"
    lines = [
        span("aaa", "gen_ai.tool.call.arguments", json.dumps({"command": "ls docs"})),
        span("aaa", "gen_ai.tool.message", "a.txt"),
        span("bbb", "gen_ai.tool.call.arguments", json.dumps({"command": "ls -R /home"})),
        span("ccc", "gen_ai.tool.message", 'File "/Users/x/anaconda3/lib/re.py" line 1'),
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    flagged = scan_spans_jsonl(path)
    assert set(flagged) == {"bbb", "ccc"}
    assert flagged["bbb"][0].field == "command"
    assert flagged["ccc"][0].field == "output"
