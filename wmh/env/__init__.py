"""The environment seam: one interface an agent loop steps against.

`Env` is the contract; `WorldModelEnv` backs it with a world model, and examples back it with
their real environments — so the same agent loop (see `wmh.env.episode.run_episode`) runs
byte-identical against either side.
"""

from wmh.env.base import Env, WorldModelEnv
from wmh.env.episode import DONE_SIGNAL, Agent, EpisodeResult, StopReason, run_episode
from wmh.env.scenarios import Scenario, scenarios_from_traces

__all__ = [
    "DONE_SIGNAL",
    "Agent",
    "Env",
    "EpisodeResult",
    "Scenario",
    "StopReason",
    "WorldModelEnv",
    "run_episode",
    "scenarios_from_traces",
]
