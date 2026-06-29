"""Tests for the CLI: command surface + build/list/play driven via CliRunner (fake provider)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wmh.cli import app
from wmh.providers.base import Completion, Message, ProviderConfig, ProviderKind, verify_via_ping

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
        # The pre-build verify guard pings through this; delegate to the shared ping so the fake
        # reports ok without hitting a real backend.
        return verify_via_ping(self)


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


def _multi_traces_file(tmp_path, trace_ids: list[str]) -> str:  # noqa: ANN001 - pytest fixture path
    lines: list[str] = []
    for i, trace_id in enumerate(trace_ids):
        span_llm = {
            "traceId": trace_id,
            "spanId": f"s{i}a",
            "name": "chat",
            "startTimeUnixNano": i * 10,
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                {"key": "gen_ai.tool.name", "value": {"stringValue": "get_user"}},
                {
                    "key": "gen_ai.tool.call.arguments",
                    "value": {"stringValue": json.dumps({"id": f"u{i}"})},
                },
                {"key": "gen_ai.prompt", "value": {"stringValue": f"look up u{i}"}},
            ],
        }
        span_tool = {
            "traceId": trace_id,
            "spanId": f"s{i}b",
            "name": "execute_tool",
            "startTimeUnixNano": i * 10 + 1,
            "attributes": [
                {"key": "gen_ai.operation.name", "value": {"stringValue": "execute_tool"}},
                {"key": "gen_ai.tool.message", "value": {"stringValue": f"found u{i}"}},
            ],
        }
        lines.append(json.dumps(span_llm))
        lines.append(json.dumps(span_tool))
    path = tmp_path / "multi_traces.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def patched_provider(monkeypatch) -> None:  # noqa: ANN001 - pytest fixture
    """Swap the real provider registry for the fake everywhere the CLI constructs one.

    Each module binds `get_provider` at its own import time (build.py for the build pipeline,
    loader.py for serve/demo/play), so patch every module-level name plus the registry the lazy
    imports read.
    """
    import sys

    import wmh.providers as providers_pkg
    import wmh.providers.registry as registry

    fake = FakeProvider()
    # `wmh.engine.__init__` rebinds the name `build` to the function, shadowing the submodule
    # attribute, so reach module objects through sys.modules rather than attribute access.
    for module_name in ("wmh.engine.build", "wmh.engine.loader"):
        monkeypatch.setattr(sys.modules[module_name], "get_provider", lambda config: fake)
    monkeypatch.setattr(providers_pkg, "get_provider", lambda config: fake)
    # The pre-build verify guard pings via verify_all/verify_embedder, which construct providers
    # through the registry's own get_provider — patch that too so the guard sees the fake.
    monkeypatch.setattr(registry, "get_provider", lambda config: fake)


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
    assert names == {"build", "list", "serve", "demo", "eval", "play"}


def test_providers_subcommand_is_registered() -> None:
    group_names = {group.name for group in app.registered_groups}
    assert "providers" in group_names


def test_bench_subcommand_is_registered() -> None:
    group_names = {group.name for group in app.registered_groups}
    assert "bench" in group_names


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


def test_play_loads_bundled_only_model(patched_provider, monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """A model that exists ONLY in the bundled dir (no writable copy) must load for play/demo.

    Regression: `_load_model` once used the writable-only `model_dir`, so a bundled model resolved
    by name but then failed to load from a nonexistent `.wmh/models/<name>` path.
    """
    import shutil

    from wmh.config.store import BUNDLED_DIR_ENV

    # Build a model into a scratch writable root, then relocate it into the bundled layout
    # (`bundled/<name>/`, no `models/` subdir) and point the store's bundled search there.
    scratch = tmp_path / "scratch"
    _build(scratch, "tau-bundled", tmp_path)
    bundled = tmp_path / "world-models"
    bundled.mkdir()
    shutil.copytree(scratch / "models" / "tau-bundled", bundled / "tau-bundled")
    monkeypatch.setenv(BUNDLED_DIR_ENV, str(bundled))

    # Empty writable root: the model is only discoverable via the bundled search path.
    empty_root = tmp_path / "empty"
    result = runner.invoke(
        app,
        ["play", "--root", str(empty_root), "--name", "tau-bundled", "--task", "look up users"],
        input='get_user {"id": "u1"}\n:quit\n',
    )
    assert result.exit_code == 0, result.output
    assert "user u1 found" in result.output


def test_build_interactive_wizard_creates_model(patched_provider, tmp_path) -> None:  # noqa: ANN001
    root = tmp_path / ".wmh"
    # --interactive forces the wizard even under CliRunner (non-TTY); feed each answer line in
    # prompt order: name, file, provider (select), model (select), region (bedrock only), budget,
    # embedder (select). The offline 'hashing' embedder skips the embed-model prompt; phi dim isn't
    # prompted. Provider/model/embedder are picked by index against their option lists.
    answers = "\n".join(
        [
            "wizard-built",
            _traces_file(tmp_path),
            "1",  # provider: bedrock
            "1",  # model: us.anthropic.claude-opus-4-8
            "us-east-1",
            "4",  # gepa budget
            "1",  # embedder: hashing
        ]
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


def test_build_aborts_when_provider_sdk_missing(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """A missing SDK must abort the build before any rollouts, with the `uv sync` extra hint.

    Regression: previously the ModuleNotFoundError was swallowed inside GEPA and the build
    "succeeded" with a useless held-out-0.0 model.
    """
    import sys

    from wmh.providers.base import VerifyResult

    appmod = sys.modules["wmh.cli.app"]
    monkeypatch.setattr(
        appmod,
        "verify_all",
        lambda configs: [
            VerifyResult(
                ok=False,
                kind=configs[0].kind,
                model=configs[0].model,
                detail="No module named 'boto3'",
            )
        ],
    )
    root = tmp_path / ".wmh"
    result = runner.invoke(
        app, ["build", "--name", "x", "--file", _traces_file(tmp_path), "--root", str(root)]
    )
    assert result.exit_code == 1
    assert "uv sync --extra bedrock" in result.output
    # Aborted before building: no artifact written.
    assert not (root / "models" / "x" / "config.toml").exists()


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


# --- bench commands ------------------------------------------------------------------------------


def _benchmark(benchmarks_root, name: str, tmp_path) -> None:  # noqa: ANN001 - pytest paths
    """Write a benchmark definition under <benchmarks_root>/<name>/ pointing at a real trace."""
    bench_dir = benchmarks_root / name
    bench_dir.mkdir(parents=True)
    trace = _traces_file(tmp_path)
    (bench_dir / "benchmark.toml").write_text(
        f'version = "1"\ntraces = ["{trace}"]\n[eval]\nseeds = [0]\n', encoding="utf-8"
    )


def _benchmark_with_trace(benchmarks_root, name: str, trace: str) -> None:  # noqa: ANN001
    bench_dir = benchmarks_root / name
    bench_dir.mkdir(parents=True)
    (bench_dir / "benchmark.toml").write_text(
        f'version = "1"\ntraces = ["{trace}"]\n[eval]\nseeds = [0]\n', encoding="utf-8"
    )


def test_bench_list_shows_definitions(tmp_path) -> None:  # noqa: ANN001
    benchmarks = tmp_path / "benchmarks"
    _benchmark(benchmarks, "tau-bench", tmp_path)
    result = runner.invoke(app, ["bench", "list", "--benchmarks", str(benchmarks)])
    assert result.exit_code == 0, result.output
    assert "tau-bench" in result.output


def test_bench_list_empty_is_friendly(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(app, ["bench", "list", "--benchmarks", str(tmp_path / "benchmarks")])
    assert result.exit_code == 0
    assert "no benchmarks" in result.output


def test_bench_leaderboard_empty_is_friendly(tmp_path) -> None:  # noqa: ANN001
    benchmarks = tmp_path / "benchmarks"
    _benchmark(benchmarks, "tau-bench", tmp_path)
    # Defined but never run: bare `wmh bench` reports no runs yet.
    result = runner.invoke(app, ["bench", "--benchmarks", str(benchmarks)])
    assert result.exit_code == 0, result.output
    assert "no benchmark runs yet" in result.output


def test_bench_run_then_leaderboard(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    import wmh.bench as bench_pkg
    from wmh.bench.runner import RolloutScore

    # Fake the scorer so the run does no LLM work; the runner + persistence + leaderboard are real.
    def fake_score(files, prompt, judge_config, **kwargs):  # noqa: ANN001, ANN003, ANN202
        return RolloutScore(fidelity_mean=0.75, fidelity_std=0.05, n_steps=4, rollouts=1)

    monkeypatch.setattr(bench_pkg, "evaluate_files_once", fake_score)

    benchmarks = tmp_path / "benchmarks"
    _benchmark(benchmarks, "tau-bench", tmp_path)

    run = runner.invoke(app, ["bench", "run", "tau-bench", "--benchmarks", str(benchmarks)])
    assert run.exit_code == 0, run.output
    assert "0.750" in run.output
    # The run persisted under the benchmark's results/ dir.
    assert list((benchmarks / "tau-bench" / "results").glob("*.json"))

    board = runner.invoke(app, ["bench", "--benchmarks", str(benchmarks)])
    assert board.exit_code == 0, board.output
    assert "tau-bench" in board.output
    assert "0.750" in board.output


def test_bench_scenario_replays_through_the_model(patched_provider, tmp_path) -> None:  # noqa: ANN001
    # Build a model, define a benchmark over the same trace, then replay the recorded scenario.
    root = tmp_path / ".wmh"
    _build(root, "scen", tmp_path)
    benchmarks = tmp_path / "benchmarks"
    _benchmark(benchmarks, "scen", tmp_path)  # benchmark name == model name (scenario default)

    result = runner.invoke(
        app,
        ["bench", "scenario", "scen", "--benchmarks", str(benchmarks), "--root", str(root)],
    )
    assert result.exit_code == 0, result.output
    assert "world model" in result.output
    # The fake provider's canned observation shows up as the live prediction, and the run finishes.
    assert "user u1 found" in result.output
    assert "done" in result.output


def test_bench_scenario_runs_multiple_replays_concurrently(
    patched_provider: object, tmp_path: Path
) -> None:
    root = tmp_path / ".wmh"
    _build(root, "scen", tmp_path)
    benchmarks = tmp_path / "benchmarks"
    trace = _multi_traces_file(tmp_path, ["a" * 32, "b" * 32])
    _benchmark_with_trace(benchmarks, "scen", trace)

    result = runner.invoke(
        app,
        [
            "bench",
            "scenario",
            "scen",
            "--scenarios",
            "2",
            "--concurrency",
            "2",
            "--benchmarks",
            str(benchmarks),
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "2 scen scenarios" in result.output
    assert "trace 0 ✓ step 0" in result.output
    assert "world model →" in result.output
    assert "world-model scenario batch" in result.output
    assert "fidelity 100%" in result.output


def test_bench_scenario_demo_batch_falls_back_to_all_traces(
    patched_provider: object, tmp_path: Path
) -> None:
    root = tmp_path / ".wmh"
    _build(root, "scen", tmp_path)
    benchmarks = tmp_path / "benchmarks"
    # "a" is held out at train_split=0.7; "c" is train. Requesting two scenarios from trace 0
    # exceeds the one-trace held-out pool but fits in the full corpus, matching the 8-way demo path.
    trace = _multi_traces_file(tmp_path, ["a" * 32, "c" * 32])
    _benchmark_with_trace(benchmarks, "scen", trace)

    result = runner.invoke(
        app,
        [
            "bench",
            "scenario",
            "scen",
            "--trace",
            "0",
            "--scenarios",
            "2",
            "--concurrency",
            "2",
            "--benchmarks",
            str(benchmarks),
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "using all 2 traces instead" in result.output
    assert "2 scen scenarios" in result.output


def test_bench_side_by_side_runs_world_model_and_real_runner(
    patched_provider: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import wmh.bench.side_by_side as side_by_side
    from wmh.bench.side_by_side import RealSandboxResult

    root = tmp_path / ".wmh"
    _build(root, "scen", tmp_path)
    benchmarks = tmp_path / "benchmarks"
    _benchmark(benchmarks, "swe-bench", tmp_path)

    def fake_run(spec, *, timeout_seconds=None):  # noqa: ANN001, ANN202
        return RealSandboxResult(
            spec=spec,
            returncode=0,
            seconds=1.2,
            stdout="REAL OUTPUT",
        )

    monkeypatch.setattr(side_by_side, "run_real_sandbox", fake_run)

    result = runner.invoke(
        app,
        [
            "bench",
            "side-by-side",
            "swe-bench",
            "--model",
            "scen",
            "--benchmarks",
            str(benchmarks),
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "side-by-side scenario summary" in result.output
    assert "mini-SWE-agent" in result.output
    assert "user u1 found" in result.output
    assert "REAL OUTPUT" in result.output


def test_bench_side_by_side_runs_multiple_scenarios_concurrently(
    patched_provider: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import wmh.bench.side_by_side as side_by_side
    from wmh.bench.side_by_side import RealSandboxResult

    root = tmp_path / ".wmh"
    _build(root, "scen", tmp_path)
    benchmarks = tmp_path / "benchmarks"
    trace = _multi_traces_file(tmp_path, ["a" * 32, "b" * 32])
    _benchmark_with_trace(benchmarks, "swe-bench", trace)
    seen: list[list[str]] = []

    def fake_run(spec, *, timeout_seconds=None):  # noqa: ANN001, ANN202
        seen.append(spec.command)
        return RealSandboxResult(spec=spec, returncode=0, seconds=1.2, stdout="REAL OUTPUT")

    monkeypatch.setattr(side_by_side, "run_real_sandbox", fake_run)

    result = runner.invoke(
        app,
        [
            "bench",
            "side-by-side",
            "swe-bench",
            "--model",
            "scen",
            "--scenarios",
            "2",
            "--concurrency",
            "2",
            "--benchmarks",
            str(benchmarks),
            "--root",
            str(root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "adding --warm" in result.output
    assert "2" in result.output
    assert len(seen) == 2
    assert all("--warm" in command for command in seen)
    assert all("--cache" in command for command in seen)
    assert any(command[command.index("--trace") + 1] == "0" for command in seen)
    assert any(command[command.index("--trace") + 1] == "1" for command in seen)


def test_bench_side_by_side_forwards_trace_pool_on_full_corpus_fallback(
    patched_provider: object, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # When the world side outgrows the held-out pool and switches to the full corpus, the real
    # runner must be told (--trace-pool all) so it resolves the SAME index against the full corpus.
    import wmh.bench.side_by_side as side_by_side
    from wmh.bench.side_by_side import RealSandboxResult

    root = tmp_path / ".wmh"
    _build(root, "scen", tmp_path)
    benchmarks = tmp_path / "benchmarks"
    # "a" is held out at split 0.7; "c" is train -> 1 held-out, 2 total. Two scenarios from trace 0
    # exceed the held-out pool but fit the full corpus, triggering the fallback.
    trace = _multi_traces_file(tmp_path, ["a" * 32, "c" * 32])
    _benchmark_with_trace(benchmarks, "swe-bench", trace)
    seen: list[list[str]] = []

    def fake_run(spec, *, timeout_seconds=None):  # noqa: ANN001, ANN202
        seen.append(spec.command)
        return RealSandboxResult(spec=spec, returncode=0, seconds=1.0, stdout="REAL OUTPUT")

    monkeypatch.setattr(side_by_side, "run_real_sandbox", fake_run)

    result = runner.invoke(
        app,
        ["bench", "side-by-side", "swe-bench", "--model", "scen", "--trace", "0",
         "--scenarios", "2", "--concurrency", "2", "--benchmarks", str(benchmarks),
         "--root", str(root)],
    )

    assert result.exit_code == 0, result.output
    assert "using all 2 traces instead" in result.output
    assert len(seen) == 2
    assert all("--trace-pool" in command for command in seen)
    assert all(command[command.index("--trace-pool") + 1] == "all" for command in seen)


def test_bench_side_by_side_rejects_zero_concurrency(
    patched_provider: object, tmp_path: Path
) -> None:
    # An explicit --concurrency 0 must be rejected, not silently treated as unset.
    root = tmp_path / ".wmh"
    _build(root, "scen", tmp_path)
    benchmarks = tmp_path / "benchmarks"
    _benchmark(benchmarks, "swe-bench", tmp_path)
    result = runner.invoke(
        app,
        ["bench", "side-by-side", "swe-bench", "--model", "scen", "--concurrency", "0",
         "--benchmarks", str(benchmarks), "--root", str(root)],
    )
    assert result.exit_code != 0
    assert "at least 1" in result.output


def test_bench_scenario_unknown_benchmark_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app, ["bench", "scenario", "ghost", "--benchmarks", str(tmp_path / "benchmarks")]
    )
    assert result.exit_code != 0
    assert not isinstance(result.exception, (FileNotFoundError, ValueError))


def test_bench_run_unknown_benchmark_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    result = runner.invoke(
        app, ["bench", "run", "ghost", "--benchmarks", str(tmp_path / "benchmarks")]
    )
    assert result.exit_code != 0
    assert not isinstance(result.exception, (FileNotFoundError, ValueError))


def test_bench_run_missing_trace_is_clean_error(tmp_path) -> None:  # noqa: ANN001
    benchmarks = tmp_path / "benchmarks"
    bench_dir = benchmarks / "tau-bench"
    bench_dir.mkdir(parents=True)
    (bench_dir / "benchmark.toml").write_text('traces = ["gone.jsonl"]\n', encoding="utf-8")
    result = runner.invoke(app, ["bench", "run", "tau-bench", "--benchmarks", str(benchmarks)])
    assert result.exit_code != 0
    assert "missing" in result.output
