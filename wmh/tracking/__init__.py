"""Run tracking: time + cost + tokens across the harness lifecycle.

Instrument at the provider boundary (`MeteredProvider`) so the world model, GEPA, and the judge are
all metered without changes to those modules. `RunTracker` aggregates `UsageEvent`s into
`UsageTotals` (priced via `wmh.tracking.pricing`) plus a wall-clock duration from an injectable
`Clock`; `RunRecord`s persist under `.wmh/runs/`.
"""

from wmh.tracking.clock import Clock, SystemClock
from wmh.tracking.metered import MeteredProvider, classify_build_call
from wmh.tracking.pricing import ModelPrice, cost_usd, price_for
from wmh.tracking.store import load_runs, save_run
from wmh.tracking.tracker import (
    Phase,
    RunRecord,
    RunTracker,
    UsageEvent,
    UsageTotals,
)

__all__ = [
    "Clock",
    "SystemClock",
    "MeteredProvider",
    "classify_build_call",
    "ModelPrice",
    "cost_usd",
    "price_for",
    "load_runs",
    "save_run",
    "Phase",
    "RunRecord",
    "RunTracker",
    "UsageEvent",
    "UsageTotals",
]
