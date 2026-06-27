"""Tests for the shared population-std helper."""

from __future__ import annotations

import pytest

from wmh.bench._stats import pop_std


def test_empty_and_single_have_no_spread() -> None:
    assert pop_std([]) == 0.0
    assert pop_std([0.9]) == 0.0


def test_population_std_of_two_values() -> None:
    # Population std of [1.0, 0.0] around mean 0.5 is 0.5.
    assert pop_std([1.0, 0.0]) == 0.5


def test_matches_known_spread() -> None:
    assert pop_std([0.2, 0.8]) == pytest.approx(0.3)
