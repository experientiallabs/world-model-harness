"""Export train-split tau episodes as neutral JSONL for the SFT arm's dataset builder.

The SFT baseline imitates the recorded tau agent. The claas-verl side
(`claas/benchmarks/wm_tau/sft.py`) turns these episodes into chat-format examples using
the SAME system prompt / message shapes / compression as the wm_tau rollout scaffold, so
the SFT row and the RL rows see byte-compatible prompts.

Leakage rule (same as pin_scenarios.py, D26): any train trace whose task text appears in
the pinned eval scenario set is dropped entirely — the policy must never train on an eval
prompt, whether as a scenario or as a recorded demonstration.

Output (~gitignored artifact root): .wmh/rl/sft_episodes.jsonl, one episode per line:
    {"trace_id", "task", "domain", "steps": [{"name", "arguments", "observation", "is_error"}]}

Run from the repo root:  uv run python examples/tau-bench/rl/export_sft_episodes.py [out.jsonl]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from wmh.config import load_config
from wmh.core.types import Trace
from wmh.engine import ingest, split_traces_3way

_HERE = Path(__file__).resolve().parent
_MODEL_DIR = _HERE.parent / "models" / "tau-bench"
_TRACES_PATH = _HERE.parent / "traces.otel.jsonl"
_EVAL_SCENARIOS = _HERE / "scenarios_eval.jsonl"
_DEFAULT_OUT = _HERE.parents[2] / ".wmh" / "rl" / "sft_episodes.jsonl"


def _trace_task(trace: Trace) -> str | None:
    for step in trace.steps:
        if step.task and step.task.strip():
            return step.task.strip()
    return None


def episodes_from_traces(traces: list[Trace], eval_tasks: set[str]) -> list[dict]:
    """Neutral episode records for SFT, dropping any trace whose task is in the eval set."""
    episodes: list[dict] = []
    for trace in traces:
        task = _trace_task(trace)
        if task is None or task in eval_tasks:
            continue
        steps = [
            {
                "name": step.action.name,
                "arguments": step.action.arguments,
                "observation": step.observation.content,
                "is_error": step.observation.is_error,
            }
            for step in trace.steps
            if step.action.name is not None
        ]
        if not steps:
            continue
        domain = trace.metadata.get("domain")
        episodes.append(
            {
                "trace_id": trace.trace_id,
                "task": task,
                "domain": domain if isinstance(domain, str) else "unknown",
                "steps": steps,
            }
        )
    return episodes


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_OUT
    config = load_config(str(_MODEL_DIR))
    traces = ingest(config, file=str(_TRACES_PATH))
    train, _val, _test = split_traces_3way(traces, 0.8, 0.1)
    eval_tasks = {
        json.loads(line)["task"] for line in _EVAL_SCENARIOS.read_text().splitlines() if line.strip()
    }
    episodes = episodes_from_traces(train, eval_tasks)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for episode in episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")
    n_steps = sum(len(e["steps"]) for e in episodes)
    dropped = len(train) - len(episodes)
    print(
        f"wrote {out}: {len(episodes)} episodes / {n_steps} steps "
        f"(dropped {dropped} of {len(train)} train traces: eval-task overlap or empty)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
