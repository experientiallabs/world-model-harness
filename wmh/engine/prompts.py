"""Prompt assembly for the world model and the demo agent.

The env prompt is the heart of the system. It composes:
  - the optimized base prompt (layer a / GEPA winner, layer b)
  - the task instruction (tau)
  - the interaction history {(s_i, a_i)}
  - the top-k retrieved demos {d_j}
  - the incoming action
into the single completion that predicts the next observation (DreamGym Eq. 4).

The actual rendering lives in `wmh.core.render` (which depends on nothing), so the serving engine
and the GEPA optimizer share one assembly — prompts are evolved against exactly what the world model
serves. This module is the engine-facing entry point: it adapts a live `Session` to that renderer.
"""

from __future__ import annotations

from wmh.core.render import build_env_prompt as _build_env_prompt
from wmh.core.render import render_demo
from wmh.core.types import Action, Session, Step

# Layer (a): the env-agnostic base prompt. GEPA (layer b) evolves a specialized version of this.
BASE_ENV_PROMPT = """You ARE the environment, not an assistant.
Given the environment state, recent interaction history, similar past examples, and the agent's
latest action, output ONLY what the environment would return in response to that action.
Predict the consequence of the action. If the action is invalid for this environment, return the
error the environment would emit. Stay consistent with the state and history. Never address the
agent or explain yourself."""


def build_env_prompt(
    base_prompt: str,
    session: Session,
    action: Action,
    demos: list[Step],
) -> tuple[str, str]:
    """Return (system, user) text for a world-model completion.

    Mirrors M_exp(R_t | {(s_i,a_i)}, {d_j}, tau): base+task -> system, history+demos+action -> user.
    Delegates to the shared renderer, supplying the session's task, state, and history.
    """
    return _build_env_prompt(
        base_prompt,
        session.task,
        session.state,
        action,
        history=session.history,
        demos=demos,
    )


def build_demo_agent_prompt(task: str, examples: list[Step]) -> str:
    """Prompt for the throwaway LLM-as-agent used by `wmh demo` (no GEPA, just examples)."""
    example_block = (
        "\n\n".join(render_demo(e) for e in examples) if examples else "(no examples)"
    )
    return (
        "You are role-playing the agent in a traced environment. Based on the task and the example "
        "interactions below, emit a SINGLE next tool call as a JSON object and nothing else:\n"
        '{"name": "<tool name>", "arguments": {<json args>}}\n\n'
        f"TASK:\n{task}\n\n"
        f"EXAMPLE INTERACTIONS:\n{example_block}\n\n"
        "Your single tool call (JSON only):"
    )
