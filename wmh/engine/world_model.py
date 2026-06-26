"""The WorldModel: a frontier LLM acting as the environment.

This is the public API agents call (in-process or via the local backend). Each `step` retrieves
similar past steps, builds the env prompt, completes it with the serving provider, and updates the
session — including the env's free-text scratchpad "database" — to stay consistent across it.
"""

from __future__ import annotations

import uuid

from wmh.core.parsing import parse_observation
from wmh.core.types import Action, EnvState, Observation, Session, Step
from wmh.engine.prompts import BASE_ENV_PROMPT, build_env_prompt
from wmh.providers.base import Embedder, Message, Provider
from wmh.retrieval import EmbeddingRetriever, Retriever


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
    def load(
        cls, artifact_dir: str, provider: Provider, embedder: Embedder | None = None
    ) -> WorldModel:
        """Construct from a built `.wmh/` artifact (optimized prompt + indexed replay buffer).

        `provider` serves the live world model (generation). `embedder` supplies phi for retrieval;
        when omitted we reconstruct the configured embedder (`embed_provider` + `embed_dim`), which
        defaults to the offline `HashingEmbedder` so loading needs no embedding credentials.
        """
        from wmh.config import ArtifactPaths, load_config
        from wmh.retrieval.embedders import get_embedder

        config = load_config(artifact_dir)
        paths = ArtifactPaths(artifact_dir)
        env_prompt = (
            paths.optimized_prompt.read_text(encoding="utf-8")
            if paths.optimized_prompt.exists()
            else BASE_ENV_PROMPT
        )
        # Reconstruct the *same* embedder the build used (provider + dim persisted in config), so
        # query vectors match the stored matrix. A caller-supplied embedder overrides.
        retriever = EmbeddingRetriever(embedder or get_embedder(config))
        if paths.index.exists():
            retriever.load(paths.index)
        return cls(provider, retriever, env_prompt=env_prompt, top_k=config.top_k)

    def new_session(self, task: str | None = None, seed_state: EnvState | None = None) -> Session:
        session = Session(id=uuid.uuid4().hex, task=task, state=seed_state or EnvState())
        self._sessions[session.id] = session
        return session

    def get_session(self, session_id: str) -> Session:
        return self._sessions[session_id]

    def sample_steps(self, n: int) -> list[Step]:
        """Return up to `n` steps from the replay buffer (used to seed the demo agent)."""
        return self._retriever.sample(n)

    def render_step_prompt(self, session_id: str, action: Action) -> str:
        """Assemble the exact (system + user) env prompt `step` would send, without calling the LLM.

        Used by `wmh demo` to display what the world model sees. Read-only: no session mutation.
        """
        session = self._sessions[session_id]
        demos = self._retriever.topk(session.state, action, self._top_k)
        system, user = build_env_prompt(self._env_prompt, session, action, demos)
        return f"{system}\n\n=== USER ===\n{user}"

    def step(self, session_id: str, action: Action) -> Observation:
        """Predict the observation for `action` and advance the session. DreamGym Eq. (4)."""
        session = self._sessions[session_id]

        # (1) retrieve top-k similar past steps conditioned on the latest state + action
        demos = self._retriever.topk(session.state, action, self._top_k)

        # (2) assemble the env prompt and (3) predict the observation
        system, user = build_env_prompt(self._env_prompt, session, action, demos)
        completion = self._provider.complete(system, [_user_message(user)])
        observation = parse_observation(completion.text)

        # (4) advance session: append step, update structured state + scratchpad, enrich buffer
        step = Step(
            action=action, observation=observation, state_before=session.state, task=session.task
        )
        session.history.append(step)
        self._update_state(session, step)
        self._retriever.add(step)
        return observation

    def _update_state(self, session: Session, step: Step) -> None:
        """Fold the step's effect into session.state (the env's free-text scratchpad "database").

        The world model emits a one-line `state_note` (carried in `observation.metadata`) describing
        what changed; we append it to the scratchpad so later steps in the session stay consistent
        (e.g. "booked r_900 for u_kath"). Structured state is left to explicit seeding/tooling.
        """
        note = step.observation.metadata.get("state_note")
        if isinstance(note, str) and note.strip():
            prefix = f"{session.state.scratchpad}\n" if session.state.scratchpad else ""
            session.state.scratchpad = f"{prefix}- {note.strip()}"


def _user_message(text: str) -> Message:
    return Message(role="user", content=text)
