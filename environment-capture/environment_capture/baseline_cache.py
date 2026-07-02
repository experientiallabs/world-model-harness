"""Load trajectories from a baseline-cache directory of frozen REAL benchmark runs.

The cache layout: ``manifest.json`` (per-task reward + the model that produced the runs),
``tasks/<task_id>.json`` (the agent-visible task), and ``traces/<task_id>.json`` (the native
trajectory: a ``messages`` list where each assistant turn carries exactly one fenced bash command
block and the following user turn carries the real ``<returncode>``/``<output>`` the environment
returned). This module parses that DATA format into Trajectories; the runs themselves were
executed for real elsewhere, so nothing here synthesizes an observation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from environment_capture.trajectory import JsonValue, StepRecord, Task, ToolCall, Trajectory

_FENCE_RE = re.compile(r"```\w*bash\s*\n(.*?)```", re.DOTALL)
_RETURNCODE_RE = re.compile(r"<returncode>(-?\d+)</returncode>")
_OUTPUT_RE = re.compile(r"<output>\n?(.*?)\n?</output>\s*\Z", re.DOTALL)


def _parse_observation(content: str, *, task_id: str) -> tuple[str, int]:
    rc_match = _RETURNCODE_RE.search(content)
    out_match = _OUTPUT_RE.search(content)
    if rc_match is None or out_match is None:
        raise ValueError(
            f"{task_id}: expected <returncode>/<output> markers in observation, got: "
            f"{content[:120]!r}. The cache trace format may have changed — update "
            f"baseline_cache.py."
        )
    return out_match.group(1), int(rc_match.group(1))


def _steps_from_messages(messages: list[dict[str, str]], *, task_id: str) -> list[StepRecord]:
    steps: list[StepRecord] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        fence = _FENCE_RE.search(message.get("content", ""))
        if fence is None:
            continue  # a final free-text turn issues no command -> no environment transition
        command = fence.group(1).strip()
        follow = messages[index + 1] if index + 1 < len(messages) else None
        if follow is None or follow.get("role") != "user":
            continue  # command with no recorded observation (run cut off) -> not a transition
        output, returncode = _parse_observation(follow.get("content", ""), task_id=task_id)
        steps.append(
            StepRecord(
                action=ToolCall(name="bash", arguments={"command": command}),
                output=output,
                is_error=returncode != 0,
            )
        )
    return steps


def load_baseline_cache(cache_dir: Path) -> list[Trajectory]:
    """Parse every task in a baseline-cache directory into Trajectories, in manifest order."""
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    model = str(manifest.get("model", ""))
    split = str(manifest.get("split", ""))

    trajectories: list[Trajectory] = []
    for entry in manifest["tasks"]:
        task_id = str(entry["task_id"])
        task_raw = json.loads((cache_dir / "tasks" / f"{task_id}.json").read_text(encoding="utf-8"))
        trace_raw = json.loads(
            (cache_dir / "traces" / f"{task_id}.json").read_text(encoding="utf-8")
        )
        metadata: dict[str, JsonValue] = {
            "source_format": "baseline-cache-v1",
            "passed": bool(entry.get("passed", False)),
            "exit_status": str(trace_raw.get("exit_status", "")),
        }
        trajectories.append(
            Trajectory(
                task=Task(
                    task_id=task_id,
                    prompt=str(task_raw.get("prompt", "")),
                    data=task_raw.get("data", {}),
                ),
                steps=_steps_from_messages(trace_raw.get("messages", []), task_id=task_id),
                final_answer=str(trace_raw.get("submission", "")),
                reward=float(entry["reward"]) if entry.get("reward") is not None else None,
                model=model,
                split=split,
                metadata=metadata,
            )
        )
    return trajectories
