"""World Model Harness — a frontier LLM acts as your agent's environment.

Public API:
    from wmh import WorldModel
    wm = WorldModel.load(".wmh", provider=...)
    session = wm.new_session(task="browse the shop")
    obs = wm.step(session.id, action)
"""

from wmh.types import (
    Action,
    ActionKind,
    EnvState,
    Observation,
    Session,
    Step,
    Trace,
)
from wmh.world_model import WorldModel

__all__ = [
    "WorldModel",
    "Action",
    "ActionKind",
    "Observation",
    "EnvState",
    "Session",
    "Step",
    "Trace",
]
