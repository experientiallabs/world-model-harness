"""The benchmark adapter contract and the capture driver.

An adapter stands up ONE real benchmark: it lists tasks per split, opens a real environment for a
task (a workspace the agent's commands actually execute in), and grades a submission
deterministically. ``run_capture`` drives an agent through every task and assembles graded
Trajectories — the only sanctioned way to produce a trace corpus (real runs, never synthesized
observations).

The ``CommandEnv.execute`` seam is deliberately the smallest possible surface: swap in an
implementation backed by a world model and the same agent loop runs against the WM instead of the
real environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from environment_capture.trajectory import StepRecord, Task, Trajectory


@dataclass(frozen=True)
class ExecResult:
    """What the environment returned for one command."""

    output: str
    returncode: int


@runtime_checkable
class CommandEnv(Protocol):
    """A live environment for one task: execute commands, then release resources."""

    def execute(self, command: str) -> ExecResult: ...

    def close(self) -> None: ...


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """One real benchmark: tasks per split, a real env per task, a deterministic grader."""

    @property
    def name(self) -> str: ...

    def tasks(self, split: str) -> list[Task]: ...

    def open_env(self, task: Task) -> CommandEnv: ...

    def grade(self, task: Task, submission: str) -> float: ...


@dataclass(frozen=True)
class AgentRun:
    """What an agent produced on one task: the real steps taken and its final answer."""

    steps: list[StepRecord]
    final_answer: str
    model: str


class CaptureAgent(Protocol):
    """Anything that can drive a CommandEnv through one task."""

    def run(self, task: Task, env: CommandEnv) -> AgentRun: ...


def run_capture(
    adapter: BenchmarkAdapter,
    agent: CaptureAgent,
    *,
    split: str,
    limit: int | None = None,
    tasks: list[Task] | None = None,
) -> list[Trajectory]:
    """Run the agent over the split's tasks against the real environment; return graded runs.

    Pass ``tasks`` to run an explicit subset (e.g. one shard of a multi-model capture); it must
    come from ``adapter.tasks(split)`` for the split label to stay truthful.
    """
    trajectories: list[Trajectory] = []
    for task in (tasks if tasks is not None else adapter.tasks(split))[:limit]:
        env = adapter.open_env(task)
        try:
            run = agent.run(task, env)
        finally:
            env.close()
        trajectories.append(
            Trajectory(
                task=task,
                steps=run.steps,
                final_answer=run.final_answer,
                reward=adapter.grade(task, run.final_answer),
                model=run.model,
                split=split,
            )
        )
    return trajectories
