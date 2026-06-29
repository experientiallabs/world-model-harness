#!/usr/bin/env python
"""Plot a concurrency scaling-law report (or several) with seaborn.

Reads the JSON `ConcurrencyScalingReport`(s) written by `scripts/run_concurrency_scaling.py` and
renders a figure: batch wall-clock vs. concurrency for each side (with mean±std error bars when the
run used `--trials`), plus the world-model speedup curve against ideal-linear. When a report has
both sides it also draws the time differential T_real(W)/T_world(W).

    uv run --extra viz python scripts/plot_concurrency_scaling.py conc_both.json --out conc.png

Pass multiple reports to overlay them (e.g. tau-bench vs. swe-bench, or world-only vs. both); each
is a separate hue. Needs the `viz` extra (seaborn/matplotlib/pandas) — the experiment itself writes
JSON with no plotting dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write a file, never try to open a window
import matplotlib.pyplot as plt  # noqa: E402 - must follow the Agg backend selection
import matplotlib.ticker as mticker  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


def _load_points(paths: list[str]) -> pd.DataFrame:
    """Flatten one report-per-level into a tidy long DataFrame for seaborn."""
    rows: list[dict[str, object]] = []
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        label = report.get("benchmark") or Path(path).stem
        for point in report["points"]:
            level = point["level"]
            # One row per (report, level, side) so wall-clock can be a single `y` with a `side` hue.
            if point.get("world_wall_mean"):
                rows.append({
                    "report": label, "level": level, "side": "world model",
                    "wall": point["world_wall_mean"], "wall_std": point["world_wall_std"],
                    "speedup": point["speedup"], "efficiency": point["efficiency"],
                    "differential": point["differential"],
                })
            if point.get("real_wall_mean"):
                rows.append({
                    "report": label, "level": level, "side": "real sandbox",
                    "wall": point["real_wall_mean"], "wall_std": point["real_wall_std"],
                    "speedup": 0.0, "efficiency": 0.0, "differential": point["differential"],
                })
    if not rows:
        raise SystemExit("no timed points found in the report(s)")
    return pd.DataFrame(rows)


def _hue(df: pd.DataFrame) -> str:
    """Use the report name as the hue when overlaying several; else the side."""
    return "report" if df["report"].nunique() > 1 else "side"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", help="ConcurrencyScalingReport JSON file(s).")
    parser.add_argument("--out", default="concurrency_scaling.png", help="Output image path.")
    parser.add_argument("--title", default="Concurrency scaling law", help="Figure title.")
    args = parser.parse_args()

    df = _load_points(args.reports)
    sns.set_theme(style="whitegrid", context="talk")
    has_diff = bool((df["differential"] > 0).any())
    ncols = 3 if has_diff else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))

    # 1) Batch wall-clock vs. concurrency, one line per side (or per report), with std error bars.
    ax = axes[0]
    hue = _hue(df)
    for key, grp in df.groupby(hue, sort=False):
        grp = grp.sort_values("level")
        ax.errorbar(
            grp["level"], grp["wall"], yerr=grp["wall_std"],
            marker="o", capsize=4, label=str(key),
        )
    ax.set(xscale="log", yscale="log", xlabel="concurrency (scenarios at once)",
           ylabel="batch wall-clock (s)", title="Batch wall-clock")
    ax.set_xticks(sorted(df["level"].unique()))
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.legend(title=hue, fontsize="small")

    # 2) World-model speedup vs. ideal linear scaling.
    ax = axes[1]
    world = df[df["side"] == "world model"]
    for key, grp in world.groupby(hue, sort=False):
        grp = grp.sort_values("level")
        ax.plot(grp["level"], grp["speedup"], marker="o", label=str(key))
    levels = sorted(df["level"].unique())
    ax.plot(levels, levels, linestyle="--", color="gray", label="ideal (linear)")
    ax.set(xlabel="concurrency", ylabel="speedup vs. W=1", title="World-model speedup")
    ax.legend(fontsize="small")

    # 3) Time differential T_real / T_world (only when both sides were timed).
    if has_diff:
        ax = axes[2]
        diff = df[(df["side"] == "real sandbox") & (df["differential"] > 0)]
        for key, grp in diff.groupby("report", sort=False):
            grp = grp.sort_values("level")
            ax.plot(grp["level"], grp["differential"], marker="o", label=str(key))
        ax.axhline(1.0, linestyle="--", color="gray", label="parity")
        ax.set(xlabel="concurrency", ylabel="T_real / T_world",
               title="Time differential (>1 = WM faster)")
        ax.legend(fontsize="small")

    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote figure -> {args.out}")


if __name__ == "__main__":
    main()
