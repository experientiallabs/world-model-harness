"""Tests for the WorldModel session lifecycle."""

from __future__ import annotations

from wmh.engine.world_model import WorldModel


def test_world_model_new_session_works() -> None:
    # new_session is concrete (no provider call); step() is what hits the stubs.
    wm = WorldModel.__new__(WorldModel)
    wm._sessions = {}
    session = WorldModel.new_session(wm, task="hi")
    assert session.id
    assert WorldModel.get_session(wm, session.id) is session
