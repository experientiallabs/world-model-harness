"""`wmh demo`: show the harness working end-to-end.

A throwaway LLM-as-agent (base prompt + a few sampled trace examples, NO GEPA) is asked to emit one
tool call. We feed that to the WorldModel and print (1) the exact env prompt sent and (2) the
predicted observation — demonstrating the loop without needing the user's real agent.
"""

from __future__ import annotations

from pydantic import BaseModel

from wmh.providers.base import Provider
from wmh.types import Action, Observation, Step
from wmh.world_model import WorldModel


class DemoResult(BaseModel):
    agent_action: Action  # what the demo agent chose to do
    env_prompt: str  # the exact prompt the world model received
    observation: Observation  # what the environment returned


def run_demo(world_model: WorldModel, agent_provider: Provider, examples: list[Step]) -> DemoResult:
    """Drive one agent->env round-trip for demonstration."""
    # 1. prompt the demo agent (base + examples) to produce a single tool call
    # 2. wm.step(...) it; capture the env prompt and observation for display
    raise NotImplementedError
