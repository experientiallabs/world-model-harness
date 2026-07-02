"""`run_episode`: the one agent-vs-environment rollout loop.

Every workstream that "runs an agent against an environment for N steps and scores the result"
uses this loop, so episode records are comparable across the world model and real backends.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from wmh.core.types import Action, EnvState, Step
from wmh.env.base import Env

# An agent signals it is finished by returning an action whose metadata-free content equals this.
DONE_SIGNAL = "<DONE>"


@runtime_checkable
class Agent(Protocol):
    """Anything that maps the episode so far to the next action.

    `act` sees the task, the initial state, and the full history of steps taken this episode.
    Return an `Action` to continue, or a MESSAGE action whose content is `DONE_SIGNAL` to stop.
    """

    def act(self, task: str | None, state: EnvState, history: list[Step]) -> Action: ...


class StopReason(StrEnum):
    AGENT_DONE = "agent_done"  # the agent returned DONE_SIGNAL
    MAX_STEPS = "max_steps"  # the step budget ran out
    ENV_ERROR = "env_error"  # env.step raised; episode recorded up to the failure


class EpisodeResult(BaseModel):
    """One completed rollout: what happened, why it stopped."""

    task: str | None = None
    steps: list[Step] = Field(default_factory=list)
    stop_reason: StopReason
    error: str | None = None  # set when stop_reason == ENV_ERROR


def run_episode(
    env_: Env,
    agent: Agent,
    task: str | None = None,
    *,
    seed_state: EnvState | None = None,
    max_steps: int = 20,
) -> EpisodeResult:
    """Roll one episode of `agent` against `env_`, bounded by `max_steps`.

    The env's `reset`/`close` bracket the episode; each turn the agent proposes an action from the
    accumulated history and the env answers with an observation. An env exception is recorded (not
    raised) so batch runs survive a flaky backend; callers inspect `stop_reason`/`error`.
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    state = env_.reset(task=task, seed_state=seed_state)
    history: list[Step] = []
    try:
        for _ in range(max_steps):
            action = agent.act(task, state, history)
            if action.content == DONE_SIGNAL:
                return EpisodeResult(task=task, steps=history, stop_reason=StopReason.AGENT_DONE)
            try:
                observation = env_.step(action)
            except Exception as exc:  # noqa: BLE001 - batch runs must survive one bad episode
                return EpisodeResult(
                    task=task,
                    steps=history,
                    stop_reason=StopReason.ENV_ERROR,
                    error=f"{type(exc).__name__}: {exc}",
                )
            history.append(
                Step(action=action, observation=observation, state_before=state, task=task)
            )
        return EpisodeResult(task=task, steps=history, stop_reason=StopReason.MAX_STEPS)
    finally:
        env_.close()
