#!/usr/bin/env python
"""Plot the 2x2 baseline grid: {base, GEPA} x {no-RAG, RAG} reconstruction fidelity.

Reads the four `wmh eval` reports produced by the grid run (each a JSON map of
`{file_key: ReplayReport}`), recomputes the step-weighted overall fidelity per cell exactly the way
`EvalReport` does, and writes a seaborn grouped bar plot plus a markdown results table.

    uv run --extra viz python scripts/plot_baseline_grid.py \
        --base-norag   benchmarks/results/grid-base-norag.json \
        --base-rag     benchmarks/results/grid-base-rag.json \
        --gepa-norag   benchmarks/results/grid-gepa-norag.json \
        --gepa-rag     benchmarks/results/grid-gepa-rag.json \
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

# A grid cell: which prompt, whether retrieval was on, and where its report lives.
PROMPT_BASE = "Base"
PROMPT_GEPA = "GEPA-optimized"
RAG_ON = "with RAG"
RAG_OFF = "no RAG"


@dataclass(frozen=True)
class Cell:
    prompt: str
    rag: str
    fidelity: float  # step-weighted mean across files
    std: float  # per-step std pooled across files
    n_steps: int


def _load_cell(path: Path, prompt: str, rag: str) -> Cell:
    """Read one `wmh eval --out` report and reduce it to a single (mean, std, n) cell.

    The report is `{file_key: {mean_score, score_std, n_steps, ...}}`. Overall fidelity is the
    step-weighted mean of per-file means; the pooled std combines the per-file variances weighted by
    step count (the same population-variance pooling `replay()` uses within a file).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        raise ValueError(f"{path} has no per-file reports")
    total = sum(int(rep["n_steps"]) for rep in data.values())
    if total == 0:
        raise ValueError(f"{path} reports zero held-out steps")
    mean = sum(rep["mean_score"] * rep["n_steps"] for rep in data.values()) / total
    # Pool variance across files: E[var] + var of the per-file means (law of total variance),
    # both step-weighted, so a single file collapses to its own score_std.
    within = sum((rep["score_std"] ** 2) * rep["n_steps"] for rep in data.values()) / total
    between = (
        sum(((rep["mean_score"] - mean) ** 2) * rep["n_steps"] for rep in data.values()) / total
    )
    return Cell(prompt=prompt, rag=rag, fidelity=mean, std=sqrt(within + between), n_steps=total)


def _markdown_table(cells: list[Cell]) -> str:
    """Render the four cells as a 2x2 markdown table (rows = prompt, cols = RAG on/off)."""
    by_key = {(c.prompt, c.rag): c for c in cells}
    lines = [
        "| Prompt | no RAG | with RAG |",
        "|---|---|---|",
    ]
    for prompt in (PROMPT_BASE, PROMPT_GEPA):
        cols = []
        for rag in (RAG_OFF, RAG_ON):
            c = by_key.get((prompt, rag))
            cols.append("—" if c is None else f"{c.fidelity:.3f} ± {c.std:.3f}")
        label = f"**{prompt}**" if prompt == PROMPT_GEPA else prompt
        lines.append(f"| {label} | {cols[0]} | {cols[1]} |")
    n = next(iter(cells)).n_steps
    lines.append("")
    lines.append(
        f"_Open-loop reconstruction fidelity (0–1), {n} held-out steps, Bedrock Opus 4.8._"
    )
    return "\n".join(lines)


def _plot(cells: list[Cell], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless: write a file, never open a window
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="talk")
    prompts = [PROMPT_BASE, PROMPT_GEPA]
    rags = [RAG_OFF, RAG_ON]
    by_key = {(c.prompt, c.rag): c for c in cells}
    palette = sns.color_palette("deep", 2)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    width = 0.36
    x = range(len(prompts))
    for j, rag in enumerate(rags):
        heights = [by_key[(p, rag)].fidelity for p in prompts]
        errs = [by_key[(p, rag)].std for p in prompts]
        positions = [i + (j - 0.5) * width for i in x]
        bars = ax.bar(
            positions,
            heights,
            width,
            yerr=errs,
            capsize=5,
            label=rag,
            color=palette[j],
            edgecolor="white",
            error_kw={"alpha": 0.6, "lw": 1.5},
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=12)

    ax.set_xticks(list(x))
    ax.set_xticklabels(prompts)
    ax.set_ylabel("Reconstruction fidelity")
    ax.set_ylim(0, 1.0)
    ax.set_title("Open-loop fidelity: prompt × retrieval (Bedrock Opus 4.8)")
    ax.legend(title="", loc="upper left", frameon=True)
    sns.despine(ax=ax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote plot -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-norag", type=Path, required=True)
    ap.add_argument("--base-rag", type=Path, required=True)
    ap.add_argument("--gepa-norag", type=Path, required=True)
    ap.add_argument("--gepa-rag", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("docs/img/baseline_grid.png"))
    ap.add_argument("--table-out", type=Path, default=None)
    args = ap.parse_args()

    cells = [
        _load_cell(args.base_norag, PROMPT_BASE, RAG_OFF),
        _load_cell(args.base_rag, PROMPT_BASE, RAG_ON),
        _load_cell(args.gepa_norag, PROMPT_GEPA, RAG_OFF),
        _load_cell(args.gepa_rag, PROMPT_GEPA, RAG_ON),
    ]
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
