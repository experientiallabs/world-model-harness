"""`wmh` CLI — ingestion UI and operator console for the harness.

Deliberately small. The lifecycle is:
    providers verify -> build -> serve / demo
`build` creates the `.wmh/` artifact directory itself, so there is no separate init step.
"""

from __future__ import annotations

import typer
from rich.console import Console

from wmh.config import ARTIFACT_DIR, HarnessConfig, load_config
from wmh.providers import ProviderConfig, ProviderKind, verify_all

app = typer.Typer(help="World Model Harness: a frontier LLM acts as your agent's environment.")
providers_app = typer.Typer(help="Manage and verify LLM providers.")
app.add_typer(providers_app, name="providers")
_console = Console()

# Module-level singleton: a typer.Argument call can't be a default inline (ruff B008).
_EVAL_FILES = typer.Argument(..., help="OTel trace files to score (one corpus each).")


@providers_app.command("verify")
def providers_verify(root: str = typer.Option(ARTIFACT_DIR, help="Artifact dir.")) -> None:
    """Ping every configured provider (Anthropic/Bedrock/Azure/OpenAI) and report status."""
    config = load_config(root)
    for result in verify_all(config.providers):
        mark = "[green]ok[/green]" if result.ok else "[red]fail[/red]"
        _console.print(f"{mark} {result.kind.value} ({result.model}) {result.detail}")


@app.command("build")
def build(
    file: str = typer.Option(None, "--file", help="Path to exported traces (OTLP-JSON / JSONL)."),
    vendor: str = typer.Option(None, "--vendor", help="Vendor name to pull traces via SDK."),
    root: str = typer.Option(ARTIFACT_DIR, help="Artifact dir to create/write."),
    provider: str = typer.Option("bedrock", "--provider", help="Provider that serves the model."),
    model: str = typer.Option("us.anthropic.claude-opus-4-8", help="Serve provider model id."),
    region: str = typer.Option(None, help="AWS region (Bedrock)."),
    gepa_budget: int = typer.Option(50, help="GEPA rollout budget."),
) -> None:
    """Ingest traces (file upload or vendor SDK pull) and build the `.wmh/` artifact.

    Creates `.wmh/` if absent, then: ingest -> normalize -> split(train/test) -> embed/index ->
    GEPA optimize -> write the artifact.
    """
    from wmh.engine.build import build as run_build
    from wmh.ingest import VendorPull

    serve_provider = ProviderKind(provider)
    config = HarnessConfig(
        providers=[ProviderConfig(kind=serve_provider, model=model, region=region)],
        serve_provider=serve_provider,
        gepa_budget=gepa_budget,
    )
    result = run_build(
        config, file=file, vendor=VendorPull() if vendor else None, root=root
    )
    _console.print(
        f"[green]built[/green] {root}: held_out_accuracy="
        f"{result.metrics.held_out_accuracy:.3f}, frontier={len(result.frontier)}, "
        f"rollouts={result.metrics.rollouts_used}"
    )


@app.command("serve")
def serve(
    port: int = typer.Option(8000, help="Port for the local backend."),
    root: str = typer.Option(ARTIFACT_DIR, help="Artifact dir to serve."),
) -> None:
    """Run the local FastAPI backend so agents can step against the world model over HTTP."""
    import uvicorn

    from wmh.serving.server import create_app

    uvicorn.run(create_app(root), host="127.0.0.1", port=port)


@app.command("eval")
def eval_(  # noqa: A001 - `eval` is the user-facing command name; the builtin isn't used here
    files: list[str] = _EVAL_FILES,
    prompt_file: str = typer.Option(None, "--prompt", help="Prompt file; default=BASE_ENV_PROMPT."),
    provider: str = typer.Option("bedrock", "--provider", help="Provider running the model."),
    model: str = typer.Option("us.anthropic.claude-opus-4-8", help="Model id."),
    region: str = typer.Option(None, help="AWS region (Bedrock)."),
    train_split: float = typer.Option(0.7, help="Train/holdout ratio per file."),
    embed_dim: int = typer.Option(512, help="phi dimensionality for the offline embedder."),
    no_rag: bool = typer.Option(False, "--no-rag", help="Disable retrieval (zero-shot replay)."),
    out: str = typer.Option(None, help="Optional path to write the full JSON report."),
) -> None:
    """Score reconstruction fidelity: replay held-out steps, judge predicted vs. real observations.

    For each trace file: split train/holdout, replay the holdout through the prompt (with leak-free
    RAG unless --no-rag), and report per-file + overall fidelity. The measurement loop behind
    iterating on the env prompt (see docs/base_prompt_iteration.md).
    """
    from pathlib import Path

    from wmh.engine.eval import evaluate_files
    from wmh.engine.prompts import BASE_ENV_PROMPT
    from wmh.optimize.judge import LLMJudge
    from wmh.providers import get_provider
    from wmh.retrieval import HashingEmbedder

    serve_provider = ProviderKind(provider)
    llm = get_provider(ProviderConfig(kind=serve_provider, model=model, region=region))
    prompt = Path(prompt_file).read_text(encoding="utf-8") if prompt_file else BASE_ENV_PROMPT
    embedder = None if no_rag else HashingEmbedder(dim=embed_dim)

    report = evaluate_files(
        [Path(f) for f in files],
        prompt,
        llm,
        LLMJudge(llm),
        embedder=embedder,
        train_split=train_split,
    )
    for name, rep in report.per_file.items():
        _console.print(f"  {name:28} {rep.summary()}")
    _console.print(
        f"[bold]OVERALL[/bold] fidelity={report.overall_fidelity:.3f} "
        f"over {report.total_steps} held-out steps"
    )
    if out:
        import json

        Path(out).write_text(
            json.dumps({n: r.model_dump() for n, r in report.per_file.items()}, indent=2),
            encoding="utf-8",
        )
        _console.print(f"wrote full report -> {out}")


@app.command("demo")
def demo(root: str = typer.Option(ARTIFACT_DIR, help="Artifact dir to demo against.")) -> None:
    """Demo the harness: an LLM agent makes a tool call vs the world model; show prompt+output."""
    from wmh.config import load_config as _load
    from wmh.engine.demo import run_demo
    from wmh.engine.world_model import WorldModel
    from wmh.providers import get_provider

    config = _load(root)
    provider = get_provider(config.serve_provider_config())
    wm = WorldModel.load(root, provider)
    # Seed the demo agent from whatever steps the index holds.
    examples = wm.sample_steps(3)
    result = run_demo(wm, provider, examples)
    _console.print(f"[bold]agent action[/bold]: {result.agent_action.model_dump()}")
    _console.print(f"[bold]env prompt[/bold]:\n{result.env_prompt}")
    _console.print(f"[bold]observation[/bold]: {result.observation.model_dump()}")


if __name__ == "__main__":
    app()
