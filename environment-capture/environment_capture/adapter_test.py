"""Tests for the BenchmarkAdapter protocol and the run_capture driver."""

from __future__ import annotations

from environment_capture.adapter import (
    AgentRun,
    BenchmarkAdapter,
    CommandEnv,
    ExecResult,
    run_capture,
)
from environment_capture.trajectory import StepRecord, Task, ToolCall


class _EchoEnv:
    """Deterministic env: echoes the command back; records close()."""

    def __init__(self) -> None:
        self.closed = False

    def execute(self, command: str) -> ExecResult:
        return ExecResult(output=f"ran:{command}", returncode=0)

    def close(self) -> None:
        self.closed = True


class _FakeAdapter:
    name = "echo-bench"

    def __init__(self) -> None:
        self.envs: list[_EchoEnv] = []

    def tasks(self, split: str) -> list[Task]:
        return [Task(task_id=f"{split}-{i}", prompt=f"say {i}", data={}) for i in range(3)]

    def open_env(self, task: Task) -> CommandEnv:
        env = _EchoEnv()
        self.envs.append(env)
        return env

    def grade(self, task: Task, submission: str) -> float:
        return 1.0 if submission == task.prompt.removeprefix("say ") else 0.0


class _OneShotAgent:
    """Runs one command through the env, then answers with the task's digit."""

    def run(self, task: Task, env: CommandEnv) -> AgentRun:
        result = env.execute(f"echo {task.task_id}")
        step = StepRecord(
            action=ToolCall(name="bash", arguments={"command": f"echo {task.task_id}"}),
            output=result.output,
            is_error=result.returncode != 0,
        )
        return AgentRun(steps=[step], final_answer=task.prompt.removeprefix("say "), model="fake")


def test_run_capture_grades_and_assembles_trajectories() -> None:
    adapter = _FakeAdapter()
    assert isinstance(adapter, BenchmarkAdapter)
    trajectories = run_capture(adapter, _OneShotAgent(), split="train")
    assert [t.task.task_id for t in trajectories] == ["train-0", "train-1", "train-2"]
    assert all(t.reward == 1.0 for t in trajectories)
    assert all(t.split == "train" for t in trajectories)
    assert trajectories[0].steps[0].output == "ran:echo train-0"
    assert all(env.closed for env in adapter.envs)


def test_run_capture_limit_and_env_closed_on_agent_error() -> None:
    adapter = _FakeAdapter()

    class _BoomAgent:
        def run(self, task: Task, env: CommandEnv) -> AgentRun:
            raise RuntimeError("agent crashed")

    try:
        run_capture(adapter, _BoomAgent(), split="train", limit=1)
    except RuntimeError:
        pass
    assert len(adapter.envs) == 1
    assert adapter.envs[0].closed  # env released even when the agent dies

    trajectories = run_capture(adapter, _OneShotAgent(), split="train", limit=2)
    assert len(trajectories) == 2
