"""Smoke tests: the skeleton imports, types instantiate, and the CLI/registry wire up.

These assert structure only — stubbed behavior raises NotImplementedError by design.
"""

from __future__ import annotations

import pytest

from wmh import Action, EnvState, Observation, Session, Step, Trace, WorldModel
from wmh.providers import ProviderConfig, ProviderKind, get_provider
from wmh.providers.base import Provider
from wmh.types import ActionKind


def test_types_instantiate() -> None:
    action = Action(kind=ActionKind.TOOL_CALL, name="cd", arguments={"path": "/tmp"})
    obs = Observation(content="", is_error=False)
    step = Step(action=action, observation=obs, state_before=EnvState(), task="poke around")
    trace = Trace(trace_id="t1", steps=[step], source="file:demo.jsonl")
    session = Session(id="s1", task="poke around")
    assert trace.steps[0].action.name == "cd"
    assert session.history == []


def test_all_four_providers_construct_and_satisfy_protocol() -> None:
    for kind in ProviderKind:
        provider = get_provider(ProviderConfig(kind=kind, model="m"))
        assert isinstance(provider, Provider)


def test_provider_verify_is_stubbed() -> None:
    provider = get_provider(ProviderConfig(kind=ProviderKind.ANTHROPIC, model="claude-opus-4-8"))
    with pytest.raises(NotImplementedError):
        provider.verify()


def test_cli_app_exposes_commands() -> None:
    from wmh.cli import app

    names = {cmd.name for cmd in app.registered_commands}
    assert {"init", "ingest", "build", "serve", "demo", "step"} <= names


def test_world_model_new_session_works() -> None:
    # new_session is concrete (no provider call); step() is what hits the stubs.
    wm = WorldModel.__new__(WorldModel)
    wm._sessions = {}  # type: ignore[attr-defined]
    session = WorldModel.new_session(wm, task="hi")
    assert session.id
    assert WorldModel.get_session(wm, session.id) is session
