"""The WorldModel: a frontier LLM acting as the environment.

This is the public API agents call (in-process or via the local backend). Each `step` retrieves
similar past steps, builds the env prompt, completes it with the serving provider, and updates the
session — including the env's free-text scratchpad "database" — to stay consistent across the session.
"""

from __future__ import annotations

import uuid

from wmh.optimize import Judge  # noqa: F401  (re-exported convenience for callers)
from wmh.prompts import BASE_ENV_PROMPT, build_env_prompt
from wmh.providers.base import Provider
from wmh.retrieval import Retriever
from wmh.types import Action, EnvState, Observation, Session, Step


class WorldModel:
    def __init__(
        self,
        provider: Provider,
        retriever: Retriever,
        env_prompt: str = BASE_ENV_PROMPT,
        top_k: int = 5,
    ) -> None:
        self._provider = provider
        self._retriever = retriever
        self._env_prompt = env_prompt
        self._top_k = top_k
        self._sessions: dict[str, Session] = {}

    @classmethod
    def load(cls, artifact_dir: str, provider: Provider) -> "WorldModel":
        """Construct from a built `.wmh/` artifact (optimized prompt + indexed replay buffer)."""
        # TODO: load config, optimized prompt, and rebuild/load the retriever index.
        raise NotImplementedError

    def new_session(self, task: str | None = None, seed_state: EnvState | None = None) -> Session:
        session = Session(id=uuid.uuid4().hex, task=task, state=seed_state or EnvState())
        self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> Session:
        return self._sessions[session_id]

    def step(self, session_id: str, action: Action) -> Observation:
        """Predict the observation for `action` and advance the session. DreamGym Eq. (4)."""
        session = self._sessions[session_id]

        # (1) retrieve top-k similar past steps conditioned on the latest state + action
        demos = self._retriever.topk(session.state, action, self._top_k)

        # (2) assemble the env prompt and (3) predict the observation
        system, user = build_env_prompt(self._env_prompt, session, action, demos)
        _completion = self._provider.complete(system, [_user_message(user)])
        observation = _parse_observation(_completion.text)  # TODO

        # (4) advance session: append step, update structured state + scratchpad, enrich buffer
        step = Step(action=action, observation=observation, state_before=session.state,
                    task=session.task)
        session.history.append(step)
        self._update_state(session, step)  # TODO: let the env write to its scratchpad "database"
        self._retriever.add(step)
        return observation

    def _update_state(self, session: Session, step: Step) -> None:
        """Fold the step's effect into session.state (structured + scratchpad)."""
        raise NotImplementedError


def _user_message(text: str):  # noqa: ANN202 - thin helper, typed at call sites
    from wmh.providers.base import Message

    return Message(role="user", content=text)


def _parse_observation(text: str) -> Observation:
    """Parse the model's raw completion into a structured Observation."""
    # TODO: define the output contract (plain text vs. tagged/JSON) and parse error/reward signals.
    raise NotImplementedError
