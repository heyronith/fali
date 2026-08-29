"""Render Figure 2 — FALI intervention game schematic."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from figure_style import COLORS, FIG_WIDTH_IN, apply_style, panel_label, save_figure  # noqa: E402


def _draw_matrix(ax) -> None:
    n = 6
    cell = 0.08
    x0, y0 = 0.08, 0.18
    labels = {
        (5, 2): ("T", COLORS["T"]),
        (5, 4): ("R", COLORS["R"]),
        (5, 0): ("R", COLORS["R"]),
        (2, 1): ("O", COLORS["O"]),
        (3, 2): ("O", COLORS["O"]),
    }
    for i in range(n):
        for j in range(i + 1):
            x = x0 + j * cell
            y = y0 + (n - 1 - i) * cell
            key = (i, j)
            if i < j or j > i:
                continue
            if i == 5 and j == 5:
                face, edge, txt = COLORS["mask"], "#999", "mask"
            elif key in labels:
                sym, edge = labels[key]
                face, txt = "white", sym
            elif i > 5 or j > i:
                face, edge, txt = COLORS["mask"], "#999", ""
            else:
                face, edge, txt = "#F0F0F0", "#AAA", ""
            if i < 5 and j > i:
                face, edge, txt = COLORS["mask"], "#999", ""
            if j > i:
                rect = Rectangle((x, y), cell, cell, facecolor=COLORS["mask"], edgecolor="#999", hatch="///", linewidth=0.5)
            else:
                rect = Rectangle((x, y), cell, cell, facecolor=face, edgecolor=edge, linewidth=0.8)
            ax.add_patch(rect)
            if txt:
                ax.text(x + cell / 2, y + cell / 2, txt, ha="center", va="center", fontsize=7, color=edge, fontweight="bold")
    ax.text(x0, y0 - 0.05, "q* row", fontsize=7)
    ax.text(x0 + 4.5 * cell, y0 + n * cell + 0.02, "structural mask — not players", fontsize=6.5, color="#666")
    for k in range(n):
        ax.add_patch(Rectangle((0.58, y0 + k * cell), 0.05, cell, facecolor=COLORS["V"], edgecolor="#333", alpha=0.25))
    ax.text(0.605, y0 + n * cell + 0.02, "V", ha="center", fontsize=8, color=COLORS["V"], fontweight="bold")


def render() -> dict:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.32))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    panel_label(ax, "(a)")
    ax.text(0.02, 0.92, "Score partition", fontsize=9, fontweight="bold")
    _draw_matrix(ax)

    ax = axes[1]
    panel_label(ax, "(b)")
    ax.text(0.5, 0.88, "A: S^A, V^A     B: S^B, V^B", ha="center", fontsize=8, family="monospace")
    steps = [
        "hybrid scores",
        "masked softmax",
        "hybrid pattern P",
        "P @ V",
        "hybrid z",
        "full candidate-z injection",
    ]
    y = 0.72
    for step in steps:
        ax.add_patch(FancyBboxPatch((0.12, y), 0.76, 0.07, boxstyle="round,pad=0.01", linewidth=0.8, edgecolor="#333", facecolor="#FAFAFA"))
        ax.text(0.5, y + 0.025, step, ha="center", fontsize=7.5)
        y -= 0.09
    ax.text(0.5, 0.12, "Example hybrid: T_B, R_A, O_B, V_A, C_B", ha="center", fontsize=7, style="italic")
    ax.text(0.5, 0.05, "U(cell) = metric(hybrid injected) − metric(candidate zero, same C)", ha="center", fontsize=7, family="monospace")

    ax = axes[2]
    panel_label(ax, "(c)")
    ax.text(0.5, 0.82, "2^5 = 32 cells", ha="center", fontsize=10, fontweight="bold", family="monospace")
    ax.text(0.5, 0.68, "SCORE = {T, R, O}", ha="center", fontsize=8, family="monospace")
    ax.text(0.5, 0.60, "VALUE = {V}", ha="center", fontsize=8, family="monospace")
    ax.text(0.5, 0.52, "CONTEXT = {C}", ha="center", fontsize=8, family="monospace")
    ax.text(0.5, 0.40, "Φ_SCORE = φ_T + φ_R + φ_O", ha="center", fontsize=8.5, family="monospace")
    ax.text(0.5, 0.30, "Φ_SCORE + φ_V + φ_C = Δ_h", ha="center", fontsize=8.5, family="monospace")
    ax.text(0.5, 0.14, "standard Owen value; 36 admissible contiguous-block permutations", ha="center", fontsize=7)
    ax.text(0.5, 0.05, "mixed cells are controlled counterfactuals", ha="center", fontsize=7, style="italic")

    return save_figure(
        fig,
        "Fig2",
        script_path=Path(__file__),
        source_files=[],
        extracted_values={"type": "deterministic_schematic"},
    )


if __name__ == "__main__":
    render()
