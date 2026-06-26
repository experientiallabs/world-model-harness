"""Terminal UX for the `wmh` CLI: guided creation/selection flows, an animated build pipeline, and
the interactive play REPL.

Everything that talks to `rich` lives here so the engine stays headless. Responsibilities:

- `run_build_wizard` interactively fills in any missing `wmh build` inputs (name, traces, provider,
  region, budget), so a bare `wmh build` becomes a guided creation flow.
- `select_model` shows a numbered picker so a user can choose which built world model to run when
  `--name` is omitted and several exist.
- `RichBuildReporter` implements `wmh.engine.reporting.BuildReporter`, turning build events into a
  guided, animated pipeline (stage lines + a live GEPA rollout progress bar) on a TTY, and into
  plain one-line-per-event output when piped (non-TTY), so logs stay legible.
- `run_play_repl` drives the human-in-the-loop demo: the user types actions, the world model
  answers, and the evolving session state (scratchpad + history) is rendered each turn.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from wmh.config import ModelInfo, validate_name
from wmh.core.types import Action, ActionKind, Session
from wmh.engine.play import PlayTurn, parse_action, play_turn
from wmh.engine.world_model import WorldModel

# A reader takes a fully-rendered prompt string and returns the user's typed line.
PromptReader = Callable[[str], str]

# Stage glyphs reused by the animated and plain reporters.
_CHECK = "[green]✓[/green]"

# Provider-specific defaults the creation wizard suggests.
_DEFAULT_MODELS: dict[str, str] = {
    "bedrock": "us.anthropic.claude-opus-4-8",
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-5.5",
    "azure_openai": "gpt-5.5",
}
_DEFAULT_REGIONS: dict[str, str] = {"bedrock": "us-east-1"}


class BuildParams(BaseModel):
    """The fully-resolved inputs for a build, as collected by the creation wizard."""

    name: str
    file: str | None = None
    vendor: str | None = None
    provider: str = "bedrock"
    model: str = "us.anthropic.claude-opus-4-8"
    region: str | None = None
    gepa_budget: int = 50


def run_build_wizard(
    console: Console, defaults: BuildParams, reader: PromptReader | None = None
) -> BuildParams:
    """Guided creation flow: prompt for each build input, pre-filled with `defaults`.

    Returns a resolved `BuildParams`. Any value already set in `defaults` (i.e. passed as a flag)
    becomes the suggested default the user can accept with Enter. Raises `ValueError` if no trace
    source (file or vendor) is provided.
    """
    ask = reader if reader is not None else (lambda text: console.input(text))
    console.print(
        Panel(
            "Let's create a world model. Press Enter to accept the [dim]default[/dim] in brackets.",
            title="[bold cyan]wmh build[/bold cyan]",
            border_style="cyan",
        )
    )

    name = _prompt_text(console, ask, "Name this world model", defaults.name)
    validate_name(name)

    file = defaults.file
    vendor = defaults.vendor
    if not file and not vendor:
        file = _prompt_text(console, ask, "Path to exported traces (OTLP-JSON / JSONL)", None)
        if not file:
            raise ValueError("a traces file is required to build (or pass --vendor)")

    provider = _prompt_text(console, ask, "Serve provider", defaults.provider)
    model_default = defaults.model or _DEFAULT_MODELS.get(provider, defaults.model)
    model = _prompt_text(console, ask, "Serve model id", model_default)
    region_default = defaults.region or _DEFAULT_REGIONS.get(provider)
    region = _prompt_text(console, ask, "Region (blank for default)", region_default) or None
    gepa_budget = _prompt_int(console, ask, "GEPA rollout budget", defaults.gepa_budget)

    return BuildParams(
        name=name,
        file=file,
        vendor=vendor,
        provider=provider,
        model=model,
        region=region,
        gepa_budget=gepa_budget,
    )


def select_model(
    console: Console, infos: list[ModelInfo], reader: PromptReader | None = None
) -> str:
    """Show a numbered picker and return the chosen model name.

    Re-prompts on invalid input. With a single model it returns that name without prompting.
    """
    if len(infos) == 1:
        return infos[0].name
    ask = reader if reader is not None else (lambda text: console.input(text))
    console.print("[bold]Select a world model:[/bold]")
    for i, info in enumerate(infos, start=1):
        acc = info.held_out_accuracy
        score = "" if acc is None else f"  [dim](held-out {acc:.2f})[/dim]"
        console.print(f"  [cyan]{i}[/cyan]. {info.name}{score}")
    while True:
        raw = ask("> ").strip()
        choice = _parse_int(raw)
        if choice is not None and 1 <= choice <= len(infos):
            return infos[choice - 1].name
        # Allow typing the name directly too.
        for info in infos:
            if raw == info.name:
                return info.name
        console.print(f"[red]pick 1-{len(infos)} or a model name[/red]")


def _prompt_text(console: Console, ask: PromptReader, label: str, default: str | None) -> str:
    suffix = f" [dim][{default}][/dim]" if default else ""
    value = ask(f"[bold]{label}[/bold]{suffix}: ").strip()
    return value or (default or "")


def _prompt_int(console: Console, ask: PromptReader, label: str, default: int) -> int:
    while True:
        raw = ask(f"[bold]{label}[/bold] [dim][{default}][/dim]: ").strip()
        if not raw:
            return default
        value = _parse_int(raw)
        if value is not None and value >= 0:
            return value
        console.print("[red]enter a non-negative whole number[/red]")


def _parse_int(raw: str) -> int | None:
    """Parse a base-10 integer, or None. Unlike `str.isdigit`, this rejects unicode digit
    characters (e.g. superscripts) that `isdigit()` accepts but `int()` rejects with ValueError."""
    try:
        return int(raw)
    except ValueError:
        return None


class RichBuildReporter:
    """A `BuildReporter` that renders the build as a guided pipeline.

    On a TTY it shows stage lines and a live progress bar for GEPA rollouts (with the running
    held-out score). When output is piped (`console.is_terminal` is false) it degrades to a single
    plain line per event — no spinners, no carriage returns — so captured logs stay readable.
    """

    def __init__(self, console: Console, model_name: str) -> None:
        self._console = console
        self._name = model_name
        self._tty = console.is_terminal
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None

    def ingest_done(self, traces: int, steps: int) -> None:
        self._stage(f"ingested {traces} traces → normalized {steps} steps")

    def split_done(self, train: int, test: int) -> None:
        self._stage(f"split {train} train / {test} held-out traces")

    def index_done(self, steps: int) -> None:
        self._stage(f"indexed {steps} steps into the replay buffer")

    def optimize_start(self, budget: int) -> None:
        self._stage(f"optimizing env prompt with GEPA (budget {budget} rollouts)")
        if self._tty and budget > 0:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TextColumn("{task.fields[score]}"),
                TimeElapsedColumn(),
                console=self._console,
                transient=True,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                "GEPA rollouts", total=budget, score="score n/a"
            )

    def rollout(self, done: int, budget: int, score: float | None) -> None:
        label = f"best held-out {score:.3f}" if score is not None else "score n/a"
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, completed=min(done, budget), score=label)
        elif not self._tty:
            # Non-TTY: emit a sparse heartbeat so long runs still show life without flooding logs.
            if done == 1 or done % 10 == 0 or done >= budget:
                self._console.print(f"  rollout {done}/{budget} ({label})")

    def optimize_done(self, held_out_accuracy: float, frontier_size: int, rollouts: int) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None
        self._stage(
            f"GEPA done: held-out {held_out_accuracy:.3f}, "
            f"{frontier_size} frontier candidates, {rollouts} rollouts used"
        )

    def _stage(self, message: str) -> None:
        self._console.print(f"{_CHECK} {message}")

    def close(self) -> None:
        """Stop the live progress bar if it is still running (e.g. the build raised mid-GEPA)."""
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None

    def __enter__(self) -> RichBuildReporter:
        return self

    def __exit__(self, *exc: object) -> None:
        # Always tear down the live Progress so an exception during the build doesn't leave a
        # spinning bar that corrupts the terminal.
        self.close()


def build_summary_panel(info: ModelInfo, root: str) -> Panel:
    """A tidy panel summarizing a freshly built world model (shown after `wmh build`)."""
    table = Table.grid(padding=(0, 2))
    table.add_column(justify="right", style="bold")
    table.add_column()
    table.add_row("name", info.name)
    table.add_row("artifact", root)
    table.add_row("serve provider", f"{info.serve_provider} ({info.serve_model})")
    if info.held_out_accuracy is not None:
        table.add_row("held-out accuracy", f"{info.held_out_accuracy:.3f}")
    if info.rollouts_used is not None:
        table.add_row("rollouts used", str(info.rollouts_used))
    if info.frontier_size is not None:
        table.add_row("frontier candidates", str(info.frontier_size))
    return Panel(
        table,
        title=f"[bold green]world model ready: {info.name}[/bold green]",
        subtitle="serve it with `wmh serve` or step into it with `wmh play`",
        border_style="green",
    )


def models_table(infos: list[ModelInfo]) -> Table:
    """A table of every built world model (for `wmh list`)."""
    table = Table(title="world models")
    table.add_column("name", style="bold")
    table.add_column("serve provider")
    table.add_column("held-out", justify="right")
    table.add_column("rollouts", justify="right")
    table.add_column("frontier", justify="right")
    for info in infos:
        table.add_row(
            info.name,
            f"{info.serve_provider} ({info.serve_model})",
            "-" if info.held_out_accuracy is None else f"{info.held_out_accuracy:.3f}",
            "-" if info.rollouts_used is None else str(info.rollouts_used),
            "-" if info.frontier_size is None else str(info.frontier_size),
        )
    return table


# --- interactive play REPL -----------------------------------------------------------------------

_PLAY_HELP = (
    "[bold]You are the agent.[/bold] Type an action and the world model answers:\n"
    '  [cyan]get_user {"id": "u1"}[/cyan]   a tool call with JSON arguments\n'
    "  [cyan]list_flights[/cyan]            a tool call with no arguments\n"
    "  [cyan]say I am stuck[/cyan]          a free-text message to the environment\n"
    "Commands: [cyan]:state[/cyan] show session state  ·  [cyan]:help[/cyan]  ·  "
    "[cyan]:quit[/cyan] (or Ctrl-D) to exit"
)


def run_play_repl(
    console: Console,
    world_model: WorldModel,
    model_name: str,
    task: str | None,
    read_line: Callable[[], str] | None = None,
) -> None:
    """Run the human-in-the-loop demo against `world_model`.

    `read_line` is an optional callable `() -> str` used to source input (injected in tests); it
    defaults to the console's prompt. The loop ends on `:quit`, EOF, or KeyboardInterrupt.
    """
    prompt = read_line if read_line is not None else (lambda: console.input("[bold]agent>[/bold] "))
    session = world_model.new_session(task=task)
    console.print(
        Panel(
            _PLAY_HELP,
            title=f"[bold]playing[/bold] {model_name}",
            subtitle=f"task: {task}" if task else "no task set",
            border_style="cyan",
        )
    )

    while True:
        try:
            line = prompt()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return
        line = line.strip()
        if not line:
            continue
        if line in {":quit", ":q", ":exit"}:
            console.print("[dim]bye[/dim]")
            return
        if line in {":help", ":h"}:
            console.print(_PLAY_HELP)
            continue
        if line == ":state":
            _render_state(console, world_model.get_session(session.id))
            continue
        _handle_action(console, world_model, session.id, line)


def _handle_action(console: Console, world_model: WorldModel, session_id: str, line: str) -> None:
    """Parse + step one typed action, rendering the observation (or a friendly error).

    A failed step (e.g. a provider/network error) is reported and swallowed so the REPL keeps the
    session alive instead of crashing the whole interactive run.
    """
    try:
        action = parse_action(line)
    except ValueError as exc:
        console.print(f"[red]parse error[/red]: {exc}")
        return
    try:
        with console.status("[dim]world model thinking…[/dim]", spinner="dots"):
            turn = play_turn(world_model, session_id, action)
    except Exception as exc:  # noqa: BLE001 - keep the REPL alive; surface the failure to the user
        console.print(f"[red]step failed[/red]: {exc}")
        return
    _render_turn(console, turn)


def _render_turn(console: Console, turn: PlayTurn) -> None:
    console.print(f"[bold cyan]→ you[/bold cyan]: {_action_text(turn.action)}")
    style = "red" if turn.observation.is_error else "green"
    label = "error" if turn.observation.is_error else "observation"
    console.print(
        Panel(
            turn.observation.content or "[dim](empty)[/dim]",
            title=f"[bold]{label}[/bold]",
            border_style=style,
        )
    )


def _render_state(console: Console, session: Session) -> None:
    scratchpad = session.state.scratchpad or "[dim](empty)[/dim]"
    body = f"[bold]task[/bold]: {session.task or '(none)'}\n"
    body += f"[bold]turns[/bold]: {len(session.history)}\n\n"
    body += f"[bold]scratchpad[/bold]:\n{scratchpad}"
    console.print(Panel(body, title="session state", border_style="blue"))


def _action_text(action: Action) -> str:
    if action.kind == ActionKind.TOOL_CALL:
        return f"{action.name}({action.arguments})"
    return f'message: "{action.content}"'
