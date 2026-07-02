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
from wmh.tracking import RunRecord


@runtime_checkable
class Env(Protocol):
    """One episode of an environment an agent steps against."""

    def reset(self, task: str | None = None, seed_state: EnvState | None = None) -> EnvState:
        """Start a fresh episode; returns the environment's LIVE state view.

        Contract: the returned `EnvState` is the env's current state object, updated in place as
        the episode advances — callers that need a point-in-time snapshot must copy it
        (`state.model_copy(deep=True)`), which is what `run_episode` does per recorded step.
        """
        ...

    def step(self, action: Action) -> Observation:
        """Apply `action` to the current episode and return the environment's response."""
        ...

    def close(self) -> None:
        """Release episode resources (sessions, containers, sim handles). Idempotent."""
        ...


class WorldModelEnv:
    """`Env` backed by a `WorldModel` session.

    Each `reset` opens a new session (ending any previous one); `step` delegates to
    `WorldModel.step`. `close` ends the session in the world model — freeing its history and
    metering — and keeps the final token/cost record available as `usage`.
    """

    def __init__(self, world_model: WorldModel) -> None:
        self._world_model = world_model
        self._session_id: str | None = None
        self._usage: RunRecord | None = None

    @property
    def session_id(self) -> str:
        if self._session_id is None:
            raise RuntimeError("WorldModelEnv has no active episode; call reset() first")
        return self._session_id

    @property
    def usage(self) -> RunRecord | None:
        """Token/cost/time of the current episode (live) or the last closed one (final)."""
        if self._session_id is not None:
            return self._world_model.session_usage(self._session_id)
        return self._usage

    def reset(self, task: str | None = None, seed_state: EnvState | None = None) -> EnvState:
        self.close()  # a leftover session would otherwise leak in the world model
        session = self._world_model.new_session(task=task, seed_state=seed_state)
        self._session_id = session.id
        return session.state

    def step(self, action: Action) -> Observation:
        return self._world_model.step(self.session_id, action)

    def close(self) -> None:
        if self._session_id is not None:
            self._usage = self._world_model.end_session(self._session_id)
            self._session_id = None
