"""Tests for the public package API surface."""

from __future__ import annotations

import wmh
from wmh.core.types import ActionKind


def test_public_api_matches_quickstart() -> None:
    # README/docstring quickstart imports ActionKind from the package root.
    assert "ActionKind" in wmh.__all__
    assert wmh.ActionKind is ActionKind
