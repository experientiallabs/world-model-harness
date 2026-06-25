"""Prompt assembly for the world model and the demo agent.

The env prompt is the heart of the system. It composes:
  - the optimized base prompt (layer a / GEPA winner, layer b)
  - the task instruction (tau)
  - the interaction history {(s_i, a_i)}
  - the top-k retrieved demos {d_j}
  - the incoming action
into the single completion that predicts the next observation (DreamGym Eq. 4).
"""

from __future__ import annotations

from wmh.core.types import Action, Session, Step

# Layer (a): the env-agnostic base prompt. GEPA (layer b) evolves a specialized version of this.
# Kept deliberately short here; the real content is tuned during `wmh build`.
BASE_ENV_PROMPT = """You ARE the environment, not an assistant.
Given the environment state, recent interaction history, similar past examples, and the agent's
latest action, output ONLY what the environment would return in response to that action.
Predict the consequence of the action. If the action is invalid for this environment, return the
error the environment would emit. Never address the agent or explain yourself."""


def build_env_prompt(
    base_prompt: str,
    session: Session,
    action: Action,
    demos: list[Step],
) -> tuple[str, str]:
    """Return (system, user) text for a world-model completion.

    Mirrors M_exp(R_t | {(s_i,a_i)}, {d_j}, tau): base+task -> system, history+demos+action -> user.
    """
    # TODO: render session.state, session.history, demos, and action into the message body.
    raise NotImplementedError


def build_demo_agent_prompt(task: str, examples: list[Step]) -> str:
    """Prompt for the throwaway LLM-as-agent used by `wmh demo` (no GEPA, just examples)."""
    # TODO: instruct the model to role-play the traced agent and emit a single tool call.
    raise NotImplementedError
