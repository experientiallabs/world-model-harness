#!/usr/bin/env python
"""Plot the baseline grid: 5 baselines x 3 corpora open-loop reconstruction fidelity.

The five baselines (the bars):
  1. Opus 4.8, base prompt           (no RAG)
  2. Opus 4.8, base prompt + RAG
  3. Opus 4.8, GEPA prompt           (no RAG)
  4. Opus 4.8, GEPA prompt + RAG
  5. Qwen-AgentWorld-35B + RAG        (trained world model, Opus judge)

across three corpora (the x-axis groups): tau2-bench, terminal-tasks, swe-bench.

Reads the `wmh eval --out` reports under a results dir, named `grid-<corpus>-<baseline>.json`
(e.g. `grid-tau2-bench-gepa-rag.json`, `grid-swe-bench-agentworld-rag.json`), each a JSON map of
`{file_key: ReplayReport}`. Recomputes the step-weighted overall fidelity per cell exactly the way
`EvalReport` does, then writes a seaborn grouped bar plot plus a markdown results table. Cells whose
report is missing are skipped (so the plot can be regenerated as runs land).

    uv run --extra viz python scripts/plot_baseline_grid.py \
        --results-dir benchmarks/results \
        --out docs/img/baseline_grid.png --table-out docs/baseline_grid_table.md

The error bar on each cell is the per-step std pooled across files (spread of fidelity across the
held-out steps), matching the ±std the README quotes — not a standard error.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

# Corpora (x-axis groups), in display order.
CORPORA = ["tau2-bench", "terminal-tasks", "swe-bench"]

# The five baselines: (file suffix, display label). Order = legend/bar order.
BASELINES = [
    ("base-norag", "Opus: base"),
    ("base-rag", "Opus: base + RAG"),
    ("gepa-norag", "Opus: GEPA"),
    ("gepa-rag", "Opus: GEPA + RAG"),
    ("agentworld-rag", "AgentWorld + RAG"),
]


@dataclass(frozen=True)
class Cell:
    corpus: str
    baseline: str  # display label
    fidelity: float  # step-weighted mean across files
    std: float  # per-step std pooled across files
    n_steps: int


def _reduce_report(data: dict[str, dict[str, float]]) -> tuple[float, float, int]:
    """Reduce a `{file_key: ReplayReport}` report to (step-weighted mean, pooled std, total steps).

    Pools variance across files via the law of total variance (within-file variance + spread of
    per-file means), step-weighted, so a single-file report collapses to its own score_std.
    """
    total = sum(int(rep["n_steps"]) for rep in data.values())
    if total == 0:
        raise ValueError("report has zero held-out steps")
    mean = sum(rep["mean_score"] * rep["n_steps"] for rep in data.values()) / total
    within = sum((rep["score_std"] ** 2) * rep["n_steps"] for rep in data.values()) / total
    between = (
        sum(((rep["mean_score"] - mean) ** 2) * rep["n_steps"] for rep in data.values()) / total
    )
    return mean, sqrt(within + between), total


def load_cells(results_dir: Path) -> list[Cell]:
    """Load every present `grid-<corpus>-<baseline>.json` under `results_dir` into a Cell."""
    cells: list[Cell] = []
    for corpus in CORPORA:
        for suffix, label in BASELINES:
            path = results_dir / f"grid-{corpus}-{suffix}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if not data:
                continue
            mean, std, n = _reduce_report(data)
            cells.append(Cell(corpus=corpus, baseline=label, fidelity=mean, std=std, n_steps=n))
    return cells


def _markdown_table(cells: list[Cell]) -> str:
    """Rows = baseline, columns = corpus; each cell is `fidelity ± std (n)`."""
    by_key = {(c.corpus, c.baseline): c for c in cells}
    present_corpora = [c for c in CORPORA if any(k[0] == c for k in by_key)]
    header = "| Baseline | " + " | ".join(present_corpora) + " |"
    sep = "|" + "---|" * (len(present_corpora) + 1)
    lines = [header, sep]
    for _suffix, label in BASELINES:
        if not any(k[1] == label for k in by_key):
            continue
        cols = []
        for corpus in present_corpora:
            c = by_key.get((corpus, label))
            cols.append("—" if c is None else f"{c.fidelity:.3f} ± {c.std:.3f}")
        bold = label.startswith("AgentWorld") or label == "Opus: GEPA + RAG"
        name = f"**{label}**" if bold else label
        lines.append(f"| {name} | " + " | ".join(cols) + " |")
    # Held-out step counts per corpus (footnote).
    counts = []
    for corpus in present_corpora:
        n = next((c.n_steps for c in cells if c.corpus == corpus), 0)
        counts.append(f"{corpus} {n}")
    lines.append("")
    lines.append(
        "_Open-loop reconstruction fidelity (0–1), all held-out turns, Bedrock Opus 4.8 judge. "
        f"Held-out steps: {', '.join(counts)}._"
    )
    return "\n".join(lines)


def _plot(cells: list[Cell], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: write a file, never open a window
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    by_key = {(c.corpus, c.baseline): c for c in cells}
    present_corpora = [c for c in CORPORA if any(k[0] == c for k in by_key)]
    labels = [lab for _s, lab in BASELINES if any(k[1] == lab for k in by_key)]
    palette = sns.color_palette("deep", len(labels))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    n = len(labels)
    group_w = 0.8
    bar_w = group_w / n
    x = range(len(present_corpora))
    for j, label in enumerate(labels):
        cells_for_label = [by_key.get((c, label)) for c in present_corpora]
        heights = [cell.fidelity if cell is not None else 0.0 for cell in cells_for_label]
        errs = [cell.std if cell is not None else 0.0 for cell in cells_for_label]
        positions = [i - group_w / 2 + bar_w * (j + 0.5) for i in x]
        bars = ax.bar(
            positions,
            heights,
            bar_w,
            yerr=errs,
            capsize=3,
            label=label,
            color=palette[j],
            edgecolor="white",
            linewidth=0.6,
            error_kw={"alpha": 0.5, "lw": 1.2},
        )
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(present_corpora)
    ax.set_ylabel("Reconstruction fidelity")
    ax.set_ylim(0, 1.0)
    ax.set_title("Open-loop world-model fidelity: 5 baselines × 3 corpora")
    ax.legend(
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote plot -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=Path("benchmarks/results"))
    ap.add_argument("--out", type=Path, default=Path("docs/img/baseline_grid.png"))
    ap.add_argument("--table-out", type=Path, default=None)
    args = ap.parse_args()

    cells = load_cells(args.results_dir)
    if not cells:
        raise SystemExit(f"no grid-*.json reports found under {args.results_dir}")
    table = _markdown_table(cells)
    print(table)
    print()
    if args.table_out is not None:
        args.table_out.parent.mkdir(parents=True, exist_ok=True)
        args.table_out.write_text(table + "\n", encoding="utf-8")
        print(f"wrote table -> {args.table_out}")
    _plot(cells, args.out)


if __name__ == "__main__":
    main()
