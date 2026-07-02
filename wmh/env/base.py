"""The `Env` protocol and its world-model backend.

An `Env` is one episode's worth of environment: `reset` starts it, `step` advances it. Real
environments (a benchmark harness, a coded oracle app, a simulator) implement the same protocol in
their example folders, which is what makes "iterate in the world model, validate in the real env"
a one-line swap instead of two agent loops.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from wmh.core.types import Action, EnvState, Observation
from wmh.engine.world_model import WorldModel


@runtime_checkable
class Env(Protocol):
    """One episode of an environment an agent steps against."""

    def reset(self, task: str | None = None, seed_state: EnvState | None = None) -> EnvState:
        """Start a fresh episode; returns the initial state the agent sees."""
        ...

    def step(self, action: Action) -> Observation:
        """Apply `action` to the current episode and return the environment's response."""
        ...

    def close(self) -> None:
        """Release episode resources (sessions, containers, sim handles). Idempotent."""
        ...


class WorldModelEnv:
    """`Env` backed by a `WorldModel` session.

    Each `reset` opens a new session; `step` delegates to `WorldModel.step`. Session usage
    (tokens/cost/time) stays available through `world_model.session_usage(env.session_id)`.
    """

    def __init__(self, world_model: WorldModel) -> None:
        self._world_model = world_model
        self._session_id: str | None = None

    @property
    def session_id(self) -> str:
        if self._session_id is None:
            raise RuntimeError("WorldModelEnv has no active episode; call reset() first")
        return self._session_id

    def reset(self, task: str | None = None, seed_state: EnvState | None = None) -> EnvState:
        session = self._world_model.new_session(task=task, seed_state=seed_state)
        self._session_id = session.id
        return session.state

    def step(self, action: Action) -> Observation:
        return self._world_model.step(self.session_id, action)

    def close(self) -> None:
        self._session_id = None
