"""Render Figure 2 — FALI intervention game schematic."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from figure_style import COLORS, FIG_WIDTH_IN, apply_style, panel_label, save_figure  # noqa: E402
from score_partition import (  # noqa: E402
    ILLUSTRATIVE_K_STAR,
    ILLUSTRATIVE_N,
    ILLUSTRATIVE_Q_STAR,
    build_score_partition,
    summarize_partition,
)


def _draw_matrix(ax) -> dict:
    grid = build_score_partition()
    summary = summarize_partition(grid)
    n = ILLUSTRATIVE_N
    cell = 0.065
    x0, y0 = 0.14, 0.20

    for q in range(n):
        for k in range(n):
            x = x0 + k * cell
            y = y0 + (n - 1 - q) * cell
            player = grid[(q, k)]
            if player == "MASK":
                rect = Rectangle(
                    (x, y),
                    cell,
                    cell,
                    facecolor=COLORS["mask"],
                    edgecolor="#999",
                    hatch="///",
                    linewidth=0.5,
                )
                ax.add_patch(rect)
            else:
                color = COLORS[player]
                rect = Rectangle((x, y), cell, cell, facecolor="white", edgecolor=color, linewidth=1.0)
                ax.add_patch(rect)
                ax.text(x + cell / 2, y + cell / 2, player, ha="center", va="center", fontsize=7, color=color, fontweight="bold")
                if q == ILLUSTRATIVE_Q_STAR and k == ILLUSTRATIVE_K_STAR:
                    ax.text(x + cell / 2, y - 0.015, "k*", ha="center", fontsize=6, color=COLORS["T"])
                if q == ILLUSTRATIVE_Q_STAR and k == n - 1:
                    ax.text(x + cell + 0.01, y + cell / 2, "q*", va="center", fontsize=6, color="#333")

    mask_x = x0 + 3.5 * cell
    mask_y = y0 + 0.5 * cell
    ax.annotate(
        "structural mask — not players",
        xy=(mask_x, mask_y),
        xytext=(mask_x + 0.12, mask_y + 0.10),
        fontsize=6.5,
        color="#666",
        arrowprops=dict(arrowstyle="-|>", color="#666", lw=0.8),
    )
    ax.text(x0 - 0.03, y0 + n * cell / 2, "query q", rotation=90, va="center", ha="center", fontsize=7)
    ax.text(x0 + n * cell / 2, y0 - 0.04, "key k", ha="center", fontsize=7)

    vx = x0 + n * cell + 0.06
    for q in range(n):
        ax.add_patch(Rectangle((vx, y0 + q * cell), 0.04, cell, facecolor=COLORS["V"], edgecolor="#333", alpha=0.25))
    ax.text(vx + 0.02, y0 + n * cell + 0.02, "V", ha="center", fontsize=8, color=COLORS["V"], fontweight="bold")

    return {
        "type": "deterministic_schematic",
        "revision": "visual/scientific-diagram correction only; no empirical recomputation",
        "partition_counts": {
            "T": summary.n_T,
            "R": summary.n_R,
            "O": summary.n_O,
            "MASK": summary.n_mask,
        },
        "final_row_players": summary.final_row,
        "q_star_q_star_player": grid[(ILLUSTRATIVE_Q_STAR, ILLUSTRATIVE_Q_STAR)],
    }


def render() -> dict:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.34))
    fig.subplots_adjust(left=0.03, right=0.98, top=0.92, bottom=0.08, wspace=0.28)
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    panel_label(ax, "(a)")
    ax.text(0.02, 0.92, "Score partition", fontsize=9, fontweight="bold")
    partition_meta = _draw_matrix(ax)

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
        extracted_values=partition_meta,
        tight=False,
        revision_note="visual/scientific-diagram correction only; no empirical recomputation",
    )


if __name__ == "__main__":
    render()
