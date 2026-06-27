"""Shared aggregation helpers for the bench package.

One definition of "population std over a sample" so the within-seed rollout std (`scoring.py`) and
the across-seed std (`results.py`) can never drift apart. Thin wrapper over stdlib
`statistics.pstdev` with the harness's "< 2 values -> 0.0" convention (a single observation has no
spread, and `pstdev` raises on an empty list).
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import pstdev


def pop_std(values: Sequence[float]) -> float:
    """Population standard deviation; 0.0 for fewer than two values."""
    return pstdev(values) if len(values) >= 2 else 0.0
