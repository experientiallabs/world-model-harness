"""Render a concurrency scaling-law report to a brand-styled matplotlib figure.

Reads the JSON `ConcurrencyScalingReport`(s) written by `wmh research concurrency --out` and draws
three panels that all compare the world model against the real sandbox: batch wall-clock per side,
each side's speedup vs. ideal-linear, and the time differential T_real(W)/T_world(W). Styling
follows the brand system (AGENTS.md rule 14): white background, near-black ink, hairline grid,
brand-palette accents, left-aligned titles — matching `scripts/plot_trace_scaling.py`.

Needs the `viz` extra (matplotlib/pandas); it is imported lazily by the CLI so the harness runtime
has no plotting dependency. Kept out of `concurrency_scaling.py` so the core experiment stays
deployment-free and fake-testable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib  # noqa: E402 - grouped with the other plotting deps below
from pydantic import BaseModel, ValidationError

from wmh.research.concurrency_scaling import ConcurrencyScalingReport

matplotlib.use("Agg")  # headless: write a file, never open a window
import matplotlib.pyplot as plt  # noqa: E402 - must follow the Agg backend selection
import matplotlib.ticker as mticker  # noqa: E402
import pandas as pd  # noqa: E402


class _PlotRow(BaseModel):
    """One tidy row for seaborn: a (report, level, side) timing point."""

    report: str
    level: int
    side: str
    wall: float
    wall_std: float
    speedup: float
    efficiency: float
    differential: float


def _load_points(paths: list[str]) -> pd.DataFrame:
    """Flatten report JSON(s) into a tidy long DataFrame (one row per report×level×side).

    Parses each file through `ConcurrencyScalingReport`, so a malformed or truncated report raises a
    clean `ValueError` (the CLI maps it to a friendly error) rather than a raw KeyError, and missing
    optional fields fall back to their model defaults.
    """
    rows: list[_PlotRow] = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        try:
            report = ConcurrencyScalingReport.model_validate_json(text)
        except ValidationError as exc:
            raise ValueError(f"{path} is not a valid concurrency-scaling report: {exc}") from exc
        label = report.benchmark or Path(path).stem
        for point in report.points:
            if point.world_wall_mean:
                rows.append(
                    _PlotRow(
                        report=label,
                        level=point.level,
                        side="world model",
                        wall=point.world_wall_mean,
                        wall_std=point.world_wall_std,
                        speedup=point.speedup,
                        efficiency=point.efficiency,
                        differential=point.differential,
                    )
                )
            if point.real_wall_mean:
                rows.append(
                    _PlotRow(
                        report=label,
                        level=point.level,
                        side="real sandbox",
                        wall=point.real_wall_mean,
                        wall_std=point.real_wall_std,
                        speedup=0.0,
                        efficiency=0.0,
                        differential=point.differential,
                    )
                )
    if not rows:
        raise ValueError("no timed points found in the report(s)")
    return pd.DataFrame([r.model_dump() for r in rows])


# Brand system (AGENTS.md rule 14): white bg, near-black ink, hairline grid, brand-palette accents.
# Ref: scripts/plot_trace_scaling.py.
_INK = "#0a0a0a"
_MUTED = "#8a8a8a"
_GRID = "#ececec"
_WORLD_COLOR = "#0070f3"  # world model — primary blue
_REAL_COLOR = "#7928ca"  # real sandbox — purple
_DIFF_COLOR = "#e00"  # differential — red
_IDEAL_COLOR = "#8a8a8a"  # ideal / parity reference — muted grey
_SIDE_COLORS = {"world model": _WORLD_COLOR, "real sandbox": _REAL_COLOR}


def _style_panel(ax: plt.Axes, levels: list[int], *, title: str, xlabel: str, ylabel: str) -> None:
    """Apply the shared brand chrome to a panel: hairline grid, no top/right spine, muted ticks."""
    ax.set_title(title, fontsize=13, color=_INK, fontweight="bold", loc="left", pad=12)
    ax.set_xlabel(xlabel, fontsize=11, color=_INK)
    ax.set_ylabel(ylabel, fontsize=11, color=_INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_GRID)
    ax.grid(axis="y", color=_GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_MUTED, labelsize=10, length=0)
    _style_level_axis(ax, levels)
    leg = ax.get_legend()
    if leg is not None:
        leg.set_frame_on(False)
        for text in leg.get_texts():
            text.set_color(_INK)


def _style_level_axis(ax: plt.Axes, levels: list[int]) -> None:
    """Log2 x-axis with plain integer concurrency ticks (1, 2, 4, ...), shared by every panel."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(levels)
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(levels[0] * 0.85, levels[-1] * 1.18)
    ax.margins(y=0.12)


def render_report(paths: list[str], out: str, *, title: str = "Concurrency scaling law") -> str:
    """Render the report JSON(s) at `paths` to an image at `out`; return `out`.

    Three panels, all comparing the SAME two sides so the world-model-vs-real-sandbox story threads
    through every one:
      1. batch wall-clock per side (absolute time),
      2. speedup vs. W=1 per side (how each parallelizes, vs ideal-linear),
      3. the time differential T_real/T_world (who is faster, and by how much).
    Fixed benchmark-agnostic styling so all benchmarks line up. When only the world side was timed
    (`--side world`), the real-sandbox series are simply absent and panel 3 says so.
    """
    df = _load_points(paths)
    has_real = bool((df["side"] == "real sandbox").any())
    has_diff = bool((df["differential"] > 0).any())
    levels = sorted(df["level"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), dpi=200)
    fig.patch.set_facecolor("white")

    def line(ax: plt.Axes, xs: list[int], ys: list[float], color: str, label: str) -> None:
        ax.plot(
            xs,
            ys,
            "-o",
            color=color,
            label=label,
            linewidth=2.2,
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.6,
            zorder=3,
        )

    # 1) Batch wall-clock vs. concurrency (log y), one line per side, mean±std band.
    ax = axes[0]
    ax.set_facecolor("white")
    for side, color in (("world model", _WORLD_COLOR), ("real sandbox", _REAL_COLOR)):
        grp = df[df["side"] == side].sort_values("level")
        if grp.empty:
            continue
        line(ax, list(grp["level"]), list(grp["wall"]), color, side)
        lo = [w - s for w, s in zip(grp["wall"], grp["wall_std"], strict=True)]
        hi = [w + s for w, s in zip(grp["wall"], grp["wall_std"], strict=True)]
        ax.fill_between(grp["level"], lo, hi, color=color, alpha=0.10, linewidth=0, zorder=2)
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=10)
    _style_panel(
        ax,
        levels,
        title="Batch wall-clock (lower = faster)",
        xlabel="concurrency (scenarios at once)",
        ylabel="batch wall-clock (s)",
    )

    # 2) Speedup vs. W=1 for BOTH sides against ideal-linear — how each side parallelizes. Speedup
    # is each side's own T(1)/T(W); the report only stores it for the world side, so derive the real
    # side's here from its wall-clocks, keeping the two-side comparison in this panel too.
    ax = axes[1]
    ax.set_facecolor("white")
    ax.plot(
        levels,
        levels,
        linestyle=(0, (4, 3)),
        color=_IDEAL_COLOR,
        linewidth=1.6,
        label="ideal (linear)",
        zorder=1,
    )
    for side, color in (("world model", _WORLD_COLOR), ("real sandbox", _REAL_COLOR)):
        grp = df[df["side"] == side].sort_values("level")
        if grp.empty:
            continue
        base = float(grp.iloc[0]["wall"])
        speedup = [base / w if w else 0.0 for w in grp["wall"]]
        line(ax, list(grp["level"]), speedup, color, side)
    ax.legend(loc="upper left", fontsize=10)
    _style_panel(
        ax,
        levels,
        title="How each side parallelizes",
        xlabel="concurrency",
        ylabel="speedup vs. W=1",
    )

    # 3) Time differential T_real / T_world (>1 = the world model is faster).
    ax = axes[2]
    ax.set_facecolor("white")
    diff = df[(df["side"] == "real sandbox") & (df["differential"] > 0)].sort_values("level")
    if has_diff and not diff.empty:
        line(ax, list(diff["level"]), list(diff["differential"]), _DIFF_COLOR, "T_real / T_world")
        for lvl, d in zip(diff["level"], diff["differential"], strict=True):
            ax.annotate(
                f"{d:.2f}×",
                (lvl, d),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
                color=_INK,
            )
        lo = min(0.95, float(diff["differential"].min()) * 0.9)
        hi = max(1.05, float(diff["differential"].max()) * 1.1)
        ax.set_ylim(lo, hi)
        ax.axhline(1.0, linestyle=(0, (4, 3)), color=_IDEAL_COLOR, linewidth=1.6, label="parity")
        ax.legend(loc="best", fontsize=10)
    else:
        msg = (
            "world-model side only\n(run `--side both` for\nthe sandbox differential)"
            if not has_real
            else "no real-sandbox timings"
        )
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color=_MUTED)
    _style_panel(
        ax,
        levels,
        title="Differential (>1 = world model faster)",
        xlabel="concurrency",
        ylabel="T_real / T_world",
    )

    fig.suptitle(title, fontsize=15, color=_INK, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


__all__ = ["render_report"]
