"""Tests for the CLI: command surface + build/list/play driven via CliRunner (fake provider)."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from wmh.cli import app
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind

runner = CliRunner()


class FakeProvider:
    """Canned world-model JSON for rollouts/steps; a fixed prompt for GEPA reflection."""

    def __init__(self) -> None:
        self.config = ProviderConfig(kind=ProviderKind.BEDROCK, model="opus")

    def complete(
        self,
        system: str,
        messages: list[Message],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Completion:
        if "improve the system prompt" in system:
            return Completion(text="IMPROVED ENV PROMPT")
        if "grade a world model" in system:
            return Completion(text='{"score": 0.5, "critique": "be more specific"}')
        return Completion(text='{"output": "user u1 found", "is_error": false}')

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def verify(self):  # noqa: ANN201
        raise NotImplementedError


def _traces_file(tmp_path) -> str:  # noqa: ANN001 - pytest fixture path
    span_llm = {
        "traceId": "a" * 32,
        "spanId": "s1",
        "name": "chat",
        "startTimeUnixNano": 1,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
            {"key": "gen_ai.tool.name", "value": {"stringValue": "get_user"}},
            {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": '{"id": "u1"}'}},
            {"key": "gen_ai.prompt", "value": {"stringValue": "look up u1"}},
        ],
    }
    span_tool = {
        "traceId": "a" * 32,
        "spanId": "s2",
        "name": "execute_tool",
        "startTimeUnixNano": 2,
        "attributes": [
            {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
            {"key": "gen_ai.tool.message", "value": {"stringValue": "found u1"}},
        ],
    }
    path = tmp_path / "traces.jsonl"
    path.write_text(json.dumps(span_llm) + "\n" + json.dumps(span_tool) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def patched_provider(monkeypatch) -> None:  # noqa: ANN001 - pytest fixture
    """Swap the real provider registry for the fake everywhere the CLI constructs one.

    `build.py` binds `get_provider` at import time, while `app.py` imports it lazily inside each
    command; we patch both the build module's bound name and the registry the lazy imports read.
    """
    import sys

    import wmh.providers as providers_pkg

    # `wmh.engine.__init__` rebinds the name `build` to the function, shadowing the submodule
    # attribute, so reach the module object through sys.modules rather than attribute access.
    build_module = sys.modules["wmh.engine.build"]

    fake = FakeProvider()
    monkeypatch.setattr(build_module, "get_provider", lambda config: fake)
    monkeypatch.setattr(providers_pkg, "get_provider", lambda config: fake)


def _build(root, name: str, tmp_path) -> None:  # noqa: ANN001 - pytest fixture paths
    result = runner.invoke(
        app,
        [
            "build",
            "--name",
            name,
            "--file",
            _traces_file(tmp_path),
            "--root",
            str(root),
            "--provider",
            "bedrock",
            "--gepa-budget",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_exposes_the_small_command_set() -> None:
    names = {cmd.name for cmd in app.registered_commands}
    assert names == {"build", "list", "serve", "demo", "play"}


def test_providers_subcommand_is_registered() -> None:
    group_names = {group.name for group in app.registered_groups}
    assert "providers" in group_names


def test_build_then_list_shows_named_model(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    _build(root, "tau2-airline", tmp_path)

    # The artifact lands under <root>/models/<name>/.
    assert (root / "models" / "tau2-airline" / "config.toml").exists()

    listed = runner.invoke(app, ["list", "--root", str(root)])
    assert listed.exit_code == 0, listed.output
    assert "tau2-airline" in listed.output


def test_list_empty_project_is_friendly(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["list", "--root", str(tmp_path / ".wmh")])
    assert result.exit_code == 0
    assert "no world models" in result.output


def test_play_repl_steps_and_quits(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    _build(root, "default", tmp_path)

    # Feed one tool call then quit; the world model's canned observation should surface.
    result = runner.invoke(
        app,
        ["play", "--root", str(root), "--task", "look up users"],
        input='get_user {"id": "u1"}\n:quit\n',
    )
    assert result.exit_code == 0, result.output
    assert "user u1 found" in result.output


def test_build_interactive_wizard_creates_model(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    # --interactive forces the wizard even under CliRunner (non-TTY); feed each answer line.
    answers = "\n".join(
        ["wizard-built", _traces_file(tmp_path), "bedrock", "opus", "us-east-1", "4"]
    )
    result = runner.invoke(
        app, ["build", "--interactive", "--root", str(root)], input=answers + "\n"
    )
    assert result.exit_code == 0, result.output
    assert (root / "models" / "wizard-built" / "config.toml").exists()


def test_build_non_interactive_without_source_errors(tmp_path) -> None:  # noqa: ANN001
    # No --file/--vendor and --no-interactive: should fail fast rather than hang on input.
    result = runner.invoke(app, ["build", "--no-interactive", "--root", str(tmp_path / ".wmh")])
    assert result.exit_code != 0


def test_play_unknown_model_errors(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["play", "--name", "nope", "--root", str(tmp_path / ".wmh")])
    assert result.exit_code != 0
    # A clean usage error, not an uncaught FileNotFoundError traceback.
    assert not isinstance(result.exception, FileNotFoundError)
    assert "nope" in result.output


def test_demo_unknown_model_is_clean_error(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    _build(root, "airline", tmp_path)
    result = runner.invoke(app, ["demo", "--name", "ghost", "--root", str(root)])
    assert result.exit_code != 0
    # Resolved through _load_model -> _resolve_name; must surface as a usage error, not a traceback.
    assert not isinstance(result.exception, (FileNotFoundError, ValueError))


def test_providers_verify_unknown_model_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app, ["providers", "verify", "--name", "ghost", "--root", str(tmp_path / ".wmh")]
    )
    assert result.exit_code != 0
    assert not isinstance(result.exception, FileNotFoundError)


def test_providers_verify_empty_project_is_friendly(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["providers", "verify", "--root", str(tmp_path / ".wmh")])
    assert result.exit_code == 0
    assert "no world models built yet" in result.output


def test_providers_verify_reports_built_model_provider(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    _build(root, "airline", tmp_path)
    result = runner.invoke(app, ["providers", "verify", "--root", str(root)])
    assert result.exit_code == 0, result.output
    # The bedrock provider configured at build time shows up in the verify report.
    assert "bedrock" in result.output
