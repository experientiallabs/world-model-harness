"""Tests for the CLI command surface."""

from __future__ import annotations

from wmh.cli import app


def test_cli_exposes_the_small_command_set() -> None:
    names = {cmd.name for cmd in app.registered_commands}
    assert names == {"build", "serve", "demo", "eval"}


def test_providers_subcommand_is_registered() -> None:
    group_names = {group.name for group in app.registered_groups}
    assert "providers" in group_names
