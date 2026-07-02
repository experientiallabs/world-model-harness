"""`wmh demo`: show the harness working end-to-end.

A throwaway LLM-as-agent (base prompt + a few sampled trace examples, NO GEPA) is asked to emit one
tool call. We feed that to the WorldModel and print (1) the exact env prompt sent and (2) the
predicted observation — demonstrating the loop without needing the user's real agent.
"""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from wmh.core.parsing import extract_json_object
from wmh.core.types import Action, ActionKind, JsonObject, Observation, Step
from wmh.engine.prompts import build_demo_agent_prompt
from wmh.engine.world_model import WorldModel
from wmh.providers.base import Message, Provider


class DemoResult(BaseModel):
    """Outputs captured from one demo-agent run."""

    agent_action: Action  # what the demo agent chose to do
    env_prompt: str  # the exact prompt the world model received
    observation: Observation  # what the environment returned


class _AgentToolCall(BaseModel):
    """Parsed tool call emitted by the demo agent."""

    name: str
    arguments: JsonObject = {}


def run_demo(world_model: WorldModel, agent_provider: Provider, examples: list[Step]) -> DemoResult:
    """Drive one agent->env round-trip for demonstration.

    A throwaway agent (base prompt + sampled examples, no GEPA) proposes one tool call; the world
    model predicts the environment's response. We capture the exact env prompt for display.
    """
    task = examples[0].task if examples and examples[0].task else "complete the task"
    action = _propose_action(agent_provider, task, examples)

    session = world_model.new_session(task=task)
    # Capture the exact prompt the world model will send (same retrieval + assembly as step()).
    env_prompt = world_model.render_step_prompt(session.id, action)
    observation = world_model.step(session.id, action)
    return DemoResult(agent_action=action, env_prompt=env_prompt, observation=observation)


def _propose_action(agent_provider: Provider, task: str, examples: list[Step]) -> Action:
    """Ask the demo agent for one tool call; fall back to a message action if it's free-form."""
    prompt = build_demo_agent_prompt(task, examples)
    completion = agent_provider.complete(
        "You role-play an agent. Reply with one tool call as JSON only.",
        [Message(role="user", content=prompt)],
        temperature=0.0,
    )
    raw = extract_json_object(completion.text)
    if raw is not None:
        try:
            call = _AgentToolCall.model_validate_json(raw)
            return Action(kind=ActionKind.TOOL_CALL, name=call.name, arguments=call.arguments)
        except ValidationError:
            pass
    return Action(kind=ActionKind.MESSAGE, content=completion.text.strip())
