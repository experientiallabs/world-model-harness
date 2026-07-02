"""Capture fresh REAL DABstep runs on Bedrock and append them to the trace corpus.

DABstep's train split is tiny (a handful of shared-context questions), so unlike a large benchmark
we do NOT round-robin shard: every model runs the FULL split (one thread per model — the pattern
for beating per-model throttling), which grows per-task coverage across models. Each trajectory's
task id is suffixed with the model + run tag (e.g. ``dab-train-3#opus48-r1``) so the deterministic
trace ids never collide across models or repeated runs. Grading uses the original task id; only the
emitted span carries the suffixed id. Raw graded trajectories are also written to ``runs/`` as JSONL
(gitignored) so a capture can be inspected without re-running.

``LocalBashEnv`` is not filesystem-sandboxed, so an agent that goes looking beyond its workspace
(``ls ~``, ``find /``) would record HOST filesystem content — not the benchmark environment — into
the corpus. Two guards keep the corpus clean: the agent is given a workspace-scoped system prompt
(its data is under ``./data/``; stay in the workspace), and any trajectory that still issued a
command escaping the workspace is dropped at emit time rather than written. (The durable fix is to
sandbox ``LocalBashEnv`` itself; until then this is capture-side containment.)

Usage (from the repo root, after fetch_data.py has pulled payments.csv):
    uv run python environment-capture/dabstep/capture.py \
        --models us.anthropic.claude-opus-4-8,us.anthropic.claude-opus-4-7 --runs 1 \
        --out environment-capture/dabstep/traces.otel.jsonl --append
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError
from environment_capture import Trajectory, run_capture, trajectory_to_spans, write_spans_jsonl
from environment_capture.agent import BedrockBashAgent
from environment_capture.benchmarks.dabstep import DabstepAdapter
from environment_capture.trajectory import Task

_HERE = Path(__file__).parent
_TASK_ATTEMPTS = 3  # a tiny split; retry transient Bedrock/network blips before giving up on a task

# Keep the agent in its workspace: the default system prompt points at a docs/ dir that doesn't
# exist here, which sends agents hunting across the host for "the data".
_WORKSPACE_SYSTEM_PROMPT = """You are an autonomous data-analysis agent. Your task's data files are
in the ./data/ directory of your current workspace (CSV/JSON plus a manual.md defining the business
rules you MUST follow to interpret the columns). Read the manual first, then analyze with python3 +
pandas. Work ONLY within the current workspace directory — never read, list, or search files outside
it (no ls ~, no find /, no absolute host paths). Use the bash tool one focused command per call and
check intermediate results. When confident, call submit with the final answer in EXACTLY the format
the question asks for (and nothing else)."""

# A command escapes the benchmark workspace if it touches an absolute host path, the home dir, or a
# parent traversal; legit dabstep commands only reference the relative ./data/ tree.
_ESCAPE_RE = re.compile(
    r"(/tmp|/Users|/home|/opt|/etc|/var|/root|/workspace|/docs|/usr|/bin|/sys|/proc"
    r"|~|\$HOME|\bcd\s+\.\.|\bfind\s+/|\bls\s+/)"
)


def _stayed_in_workspace(trajectory: Trajectory) -> bool:
    """True if every command stayed inside the workspace (no host-filesystem escape)."""
    return not any(
        _ESCAPE_RE.search(str(step.action.arguments.get("command", "")))
        for step in trajectory.steps
    )


def _model_tag(model_id: str) -> str:
    """Short alphanumeric id for a Bedrock model, used to keep suffixed task ids unique."""
    tail = model_id.split("claude-")[-1]
    return re.sub(r"[^a-z0-9]", "", tail)


def _suffix_task_id(trajectory: Trajectory, tag: str) -> Trajectory:
    """Re-key a graded trajectory's task id with a run suffix (after grading, before emission)."""
    task = replace(trajectory.task, task_id=f"{trajectory.task.task_id}#{tag}")
    return replace(trajectory, task=task)


def _capture_model(
    adapter: DabstepAdapter,
    model_id: str,
    tasks: list[Task],
    runs: int,
    run_start: int,
    max_steps: int,
) -> list[Trajectory]:
    """Run one model over every task, isolating each task so a transient failure loses only it.

    ``run_capture`` has no per-task fault isolation, and the models run concurrently under one
    executor — so one uncaught Bedrock/network error would discard every model's completed work.
    Each task is instead driven on its own with a few retries; a task that keeps failing is skipped
    with a warning rather than aborting the capture. Runs are numbered from ``run_start`` so a
    top-up capture (``--run-start 2``) never reuses an earlier run's suffix (and thus trace id).
    """
    agent = BedrockBashAgent(model_id, max_steps=max_steps, system_prompt=_WORKSPACE_SYSTEM_PROMPT)
    tag = _model_tag(model_id)
    captured: list[Trajectory] = []
    for run_index in range(run_start, run_start + runs):
        for task in tasks:
            trajectory = _capture_task(adapter, agent, task, tag, run_index)
            if trajectory is not None:
                captured.append(_suffix_task_id(trajectory, f"{tag}-r{run_index}"))
    return captured


def _capture_task(
    adapter: DabstepAdapter,
    agent: BedrockBashAgent,
    task: Task,
    tag: str,
    run_index: int,
) -> Trajectory | None:
    """Drive one task with retries; return its graded trajectory, or None if it keeps failing."""
    for attempt in range(1, _TASK_ATTEMPTS + 1):
        try:
            return run_capture(adapter, agent, split="train", tasks=[task])[0]
        except (BotoCoreError, ClientError) as error:
            print(f"  [{tag} r{run_index}] {task.task_id} attempt {attempt} failed: {error}")
    print(f"  [{tag} r{run_index}] {task.task_id} giving up after {_TASK_ATTEMPTS} attempts")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of tasks")
    parser.add_argument("--skip", type=int, default=0, help="Skip the first N tasks")
    parser.add_argument(
        "--models",
        default="us.anthropic.claude-opus-4-8",
        help="Comma-separated Bedrock model ids; each runs the FULL split",
    )
    parser.add_argument("--runs", type=int, default=1, help="Runs per model over the split")
    parser.add_argument(
        "--run-start",
        type=int,
        default=1,
        help="First run index (bump to top up a corpus without reusing earlier run suffixes)",
    )
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--out", default=str(_HERE / "traces.otel.jsonl"))
    parser.add_argument("--append", action="store_true", help="Append to --out (default: refuse)")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and not args.append:
        raise SystemExit(f"{out} exists; pass --append to extend it")
    if args.split == "test":
        raise SystemExit(
            "refusing to capture the test split into a corpus: the hidden test split must stay "
            "out of world-model training data"
        )

    adapter = DabstepAdapter(data_root=_HERE)
    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = adapter.tasks(args.split)[args.skip :]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    started = time.time()
    with ThreadPoolExecutor(max_workers=len(model_ids)) as pool:
        model_results = list(
            pool.map(
                lambda model_id: _capture_model(
                    adapter, model_id, tasks, args.runs, args.run_start, args.max_steps
                ),
                model_ids,
            )
        )
    trajectories = [t for result in model_results for t in result]

    runs_dir = _HERE / "runs"
    runs_dir.mkdir(exist_ok=True)
    raw_path = runs_dir / f"capture-{int(started)}.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw:
        for trajectory in trajectories:
            raw.write(json.dumps(asdict(trajectory), ensure_ascii=False) + "\n")

    with_steps = [t for t in trajectories if t.steps]
    escaped = [t for t in with_steps if not _stayed_in_workspace(t)]
    for trajectory in escaped:
        print(
            f"[drop] {trajectory.task.task_id}: escaped the workspace (host filesystem content) — "
            f"not emitted",
            file=sys.stderr,
        )
    kept = [t for t in with_steps if _stayed_in_workspace(t)]
    n_spans = 0
    for index, trajectory in enumerate(kept):
        spans = trajectory_to_spans(trajectory, benchmark="dabstep")
        n_spans += write_spans_jsonl(spans, out, append=args.append or index > 0)

    rewards = [t.reward or 0.0 for t in kept]
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    n_zero_step = len(trajectories) - len(with_steps)
    print(
        f"captured {len(trajectories)} runs; emitted {len(kept)} workspace-contained "
        f"({sum(len(t.steps) for t in kept)} steps, mean reward {mean_reward:.3f}); "
        f"dropped {len(escaped)} escaped + {n_zero_step} zero-step "
        f"in {time.time() - started:.0f}s -> {out} (raw: {raw_path})"
    )


if __name__ == "__main__":
    main()
