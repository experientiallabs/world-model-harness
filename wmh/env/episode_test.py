"""Tests for run_episode: stop reasons, history threading, env lifecycle."""

from __future__ import annotations

import pytest

from wmh.core.types import Action, ActionKind, EnvState, Observation, Step
from wmh.env import EpisodeResult, StopReason, run_episode
from wmh.env.episode import DONE_SIGNAL


class ScriptedEnv:
    """Deterministic Env: replies `obs {n}`; records lifecycle calls."""

    def __init__(self, fail_on_step: int | None = None) -> None:
        self.reset_calls = 0
        self.closed = False
        self._n = 0
        self._fail_on_step = fail_on_step

    def reset(self, task: str | None = None, seed_state: EnvState | None = None) -> EnvState:
        self.reset_calls += 1
        return seed_state or EnvState(scratchpad="start")

    def step(self, action: Action) -> Observation:
        self._n += 1
        if self._fail_on_step is not None and self._n >= self._fail_on_step:
            raise ConnectionError("backend went away")
        return Observation(content=f"obs {self._n}")

    def close(self) -> None:
        self.closed = True


class ScriptedAgent:
    """Emits tool calls until `stop_after` steps have been taken, then signals done."""

    def __init__(self, stop_after: int | None = None) -> None:
        self._stop_after = stop_after
        self.seen_history_lengths: list[int] = []

    def act(self, task: str | None, state: EnvState, history: list[Step]) -> Action:
        self.seen_history_lengths.append(len(history))
        if self._stop_after is not None and len(history) >= self._stop_after:
            return Action(kind=ActionKind.MESSAGE, content=DONE_SIGNAL)
        return Action(kind=ActionKind.TOOL_CALL, name="poke", arguments={"turn": len(history)})


def test_episode_stops_when_agent_is_done() -> None:
    env = ScriptedEnv()
    agent = ScriptedAgent(stop_after=2)
    result = run_episode(env, agent, task="t", max_steps=10)

    assert isinstance(result, EpisodeResult)
    assert result.stop_reason == StopReason.AGENT_DONE
    assert [s.observation.content for s in result.steps] == ["obs 1", "obs 2"]
    # The agent saw the growing history each turn: 0, 1, then 2 steps.
    assert agent.seen_history_lengths == [0, 1, 2]
    assert env.closed


def test_episode_hits_max_steps() -> None:
    result = run_episode(ScriptedEnv(), ScriptedAgent(), task="t", max_steps=3)
    assert result.stop_reason == StopReason.MAX_STEPS
    assert len(result.steps) == 3


def test_episode_records_env_error_instead_of_raising() -> None:
    env = ScriptedEnv(fail_on_step=2)
    result = run_episode(env, ScriptedAgent(), max_steps=5)
    assert result.stop_reason == StopReason.ENV_ERROR
    assert result.error is not None and "ConnectionError" in result.error
    assert len(result.steps) == 1  # everything before the failure is kept
    assert env.closed


def test_episode_threads_task_and_state_into_steps() -> None:
    result = run_episode(
        ScriptedEnv(),
        ScriptedAgent(stop_after=1),
        task="find alice",
        seed_state=EnvState(scratchpad="seeded"),
        max_steps=5,
    )
    step = result.steps[0]
    assert step.task == "find alice"
    assert step.state_before.scratchpad == "seeded"


def test_episode_rejects_nonpositive_budget() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        run_episode(ScriptedEnv(), ScriptedAgent(), max_steps=0)
