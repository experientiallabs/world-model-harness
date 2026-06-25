"""The world-model engine: prompt assembly, the WorldModel, the build pipeline, and the demo."""

from wmh.engine.build import build, ingest, split_traces
from wmh.engine.demo import DemoResult, run_demo
from wmh.engine.world_model import WorldModel

__all__ = ["build", "ingest", "split_traces", "DemoResult", "run_demo", "WorldModel"]
