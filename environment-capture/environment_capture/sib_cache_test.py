"""Tests for loading SIB baseline-cache trajectories (data-only reuse, DECISIONS.md D21)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from environment_capture.sib_cache import load_sib_cache

_ASSISTANT_0 = "I'll list the docs first.\n```sib_bash\nls docs && grep -in capex docs/*.txt\n```"
_USER_0 = "<returncode>0</returncode>\n<output>\na.txt\nb.txt\n</output>"
_ASSISTANT_1 = "Submitting.\n```sib_bash\nprintf 'SIB_SUBMIT\\n$1577.00\\n'\n```"
_USER_1 = "<returncode>0</returncode>\n<output>\nSIB_SUBMIT\n$1577.00\n</output>"


def _write_cache(root: Path) -> Path:
    (root / "tasks").mkdir(parents=True)
    (root / "traces").mkdir()
    manifest = {
        "benchmark": "financebench",
        "split": "train",
        "model": "gpt-5.4",
        "n": 2,
        "mean_reward": 0.5,
        "pass_rate": 0.5,
        "tasks": [
            {"task_id": "fb-train-0", "passed": True, "reward": 1.0, "exit_status": "Submitted"},
            {"task_id": "fb-train-1", "passed": False, "reward": 0.0, "exit_status": "Submitted"},
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    for task_id, rc in (("fb-train-0", 0), ("fb-train-1", 1)):
        task_payload = {
            "task_id": task_id,
            "prompt": f"Question for {task_id}?",
            "data": {"stratum": "easy"},
        }
        (root / "tasks" / f"{task_id}.json").write_text(json.dumps(task_payload))
        user_0 = _USER_0 if rc == 0 else f"<returncode>{rc}</returncode>\n<output>\nboom\n</output>"
        trace = {
            "submission": "$1577.00",
            "exit_status": "Submitted",
            "steps": 2,
            "cost_usd": 0.01,
            "tokens": 1000,
            "messages": [
                {"role": "system", "content": "You are an agent."},
                {"role": "user", "content": f"Question for {task_id}?"},
                {"role": "assistant", "content": _ASSISTANT_0},
                {"role": "user", "content": user_0},
                {"role": "assistant", "content": _ASSISTANT_1},
                {"role": "user", "content": _USER_1},
            ],
        }
        (root / "traces" / f"{task_id}.json").write_text(json.dumps(trace))
    return root


def test_load_sib_cache_parses_commands_and_observations(tmp_path: Path) -> None:
    trajectories = load_sib_cache(_write_cache(tmp_path))
    assert [t.task.task_id for t in trajectories] == ["fb-train-0", "fb-train-1"]

    ok = trajectories[0]
    assert ok.model == "gpt-5.4"
    assert ok.split == "train"
    assert ok.reward == 1.0
    assert ok.final_answer == "$1577.00"
    assert ok.metadata["passed"] is True
    assert ok.task.prompt == "Question for fb-train-0?"
    assert len(ok.steps) == 2
    first = ok.steps[0]
    assert first.action.name == "bash"
    assert first.action.arguments == {"command": "ls docs && grep -in capex docs/*.txt"}
    assert first.output == "a.txt\nb.txt"
    assert first.is_error is False

    failed = trajectories[1]
    assert failed.reward == 0.0
    assert failed.steps[0].is_error is True
    assert failed.steps[0].output == "boom"


def test_load_sib_cache_rejects_malformed_observation(tmp_path: Path) -> None:
    root = _write_cache(tmp_path)
    trace_path = root / "traces" / "fb-train-0.json"
    trace = json.loads(trace_path.read_text())
    trace["messages"][3]["content"] = "no markers here"
    trace_path.write_text(json.dumps(trace))
    with pytest.raises(ValueError, match="fb-train-0"):
        load_sib_cache(root)
