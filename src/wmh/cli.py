"""`wmh` CLI — the UI for ingestion and the operator console for the harness.

Commands map onto the lifecycle in DESIGN.md:
    init -> providers verify -> ingest -> build -> serve / demo / step
"""

from __future__ import annotations

import typer

app = typer.Typer(help="World Model Harness: a frontier LLM acts as your agent's environment.")
providers_app = typer.Typer(help="Manage and verify LLM providers.")
app.add_typer(providers_app, name="providers")


@app.command("init")
def init() -> None:
    """Scaffold a `.wmh/` project and write a starter config."""
    raise NotImplementedError


@providers_app.command("verify")
def providers_verify() -> None:
    """Ping every configured provider (Anthropic/Bedrock/Azure/OpenAI) and report status."""
    raise NotImplementedError


@app.command("ingest")
def ingest(
    file: str = typer.Option(None, "--file", help="Path to exported traces (OTLP-JSON / JSONL)."),
    vendor: str = typer.Option(None, "--vendor", help="Vendor name to pull traces via SDK."),
) -> None:
    """Ingest traces from a file upload or a vendor SDK pull into `.wmh/traces/`."""
    raise NotImplementedError


@app.command("build")
def build() -> None:
    """Normalize -> split -> embed/index -> GEPA optimize -> write the `.wmh/` artifact."""
    raise NotImplementedError


@app.command("serve")
def serve(port: int = typer.Option(8000, help="Port for the local backend.")) -> None:
    """Run the local FastAPI backend so agents can step against the world model over HTTP."""
    raise NotImplementedError


@app.command("demo")
def demo() -> None:
    """Demo the harness: an LLM agent makes a tool call vs the world model; show prompt+output."""
    raise NotImplementedError


@app.command("step")
def step(
    session: str = typer.Option(..., "--session", help="Session id."),
    tool: str = typer.Option(..., "--tool", help="Tool name to call."),
    args: str = typer.Option("{}", "--args", help="JSON tool arguments."),
) -> None:
    """Issue a single tool-call step from the CLI (debugging convenience)."""
    raise NotImplementedError


if __name__ == "__main__":
    app()
