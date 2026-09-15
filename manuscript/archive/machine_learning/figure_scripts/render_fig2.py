"""Render Figure 2 — FALI intervention game schematic."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from figure_style import (  # noqa: E402
    COLORS,
    FIG_WIDTH_IN,
    MIN_FONT_PT,
    PREF_FONT_PT,
    apply_style,
    fig_height_in,
    panel_label,
    save_figure,
)
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
    cell = 0.075
    x0, y0 = 0.10, 0.18

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
                    linewidth=0.6,
                )
                ax.add_patch(rect)
            else:
                color = COLORS[player]
                rect = Rectangle((x, y), cell, cell, facecolor="white", edgecolor=color, linewidth=1.0)
                ax.add_patch(rect)
                ax.text(
                    x + cell / 2,
                    y + cell / 2,
                    player,
                    ha="center",
                    va="center",
                    fontsize=MIN_FONT_PT,
                    color=color,
                    fontweight="bold",
                )
                if q == ILLUSTRATIVE_Q_STAR and k == ILLUSTRATIVE_K_STAR:
                    ax.text(x + cell / 2, y - 0.018, "k*", ha="center", fontsize=MIN_FONT_PT, color=COLORS["T"])
                if q == ILLUSTRATIVE_Q_STAR and k == n - 1:
                    ax.text(x + cell + 0.012, y + cell / 2, "q*", va="center", fontsize=MIN_FONT_PT, color="#333")

    mask_x = x0 + 3.5 * cell
    mask_y = y0 + 0.5 * cell
    ax.annotate(
        "structural mask",
        xy=(mask_x, mask_y),
        xytext=(mask_x + 0.16, mask_y + 0.12),
        fontsize=MIN_FONT_PT,
        color="#666",
        arrowprops=dict(arrowstyle="-|>", color="#666", lw=0.8),
    )
    ax.text(x0 - 0.04, y0 + n * cell / 2, "query q", rotation=90, va="center", ha="center", fontsize=PREF_FONT_PT)
    ax.text(x0 + n * cell / 2, y0 - 0.05, "key k", ha="center", fontsize=PREF_FONT_PT)

    vx = x0 + n * cell + 0.08
    for q in range(n):
        ax.add_patch(Rectangle((vx, y0 + q * cell), 0.045, cell, facecolor=COLORS["V"], edgecolor="#333", alpha=0.25))
    ax.text(vx + 0.022, y0 + n * cell + 0.025, "V", ha="center", fontsize=PREF_FONT_PT, color=COLORS["V"], fontweight="bold")

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
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height_in("Fig2")))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 0.95], width_ratios=[0.48, 0.52], hspace=0.30, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    for ax in (ax_a, ax_b, ax_c):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    panel_label(ax_a, "(a)")
    ax_a.text(0.02, 0.94, "Score partition", fontsize=PREF_FONT_PT, fontweight="bold")
    partition_meta = _draw_matrix(ax_a)

    panel_label(ax_b, "(b)")
    ax_b.text(0.5, 0.90, "T_B  R_A  O_B  V_A  C_B", ha="center", fontsize=PREF_FONT_PT, family="monospace")
    steps = [
        "Hybrid scores",
        "Masked softmax",
        "Pattern P",
        "P @ V",
        "Candidate z",
        "Inject full z",
    ]
    y = 0.74
    for step in steps:
        ax_b.add_patch(
            FancyBboxPatch(
                (0.10, y),
                0.80,
                0.075,
                boxstyle="round,pad=0.012",
                linewidth=0.8,
                edgecolor="#333",
                facecolor="#FAFAFA",
            )
        )
        ax_b.text(0.5, y + 0.028, step, ha="center", fontsize=MIN_FONT_PT)
        if y > 0.20:
            ax_b.add_patch(
                FancyArrowPatch(
                    (0.5, y),
                    (0.5, y - 0.02),
                    transform=ax_b.transData,
                    arrowstyle="-|>",
                    mutation_scale=10,
                    linewidth=0.8,
                    color="#666",
                )
            )
        y -= 0.095
    ax_b.text(0.5, 0.10, "U(cell) = metric(hybrid candidate)", ha="center", fontsize=MIN_FONT_PT, family="monospace")
    ax_b.text(0.5, 0.04, "          − metric(candidate zero, same C)", ha="center", fontsize=MIN_FONT_PT, family="monospace")

    panel_label(ax_c, "(c)")
    ax_c.text(0.12, 0.72, "32 hybrid cells", ha="center", fontsize=PREF_FONT_PT, fontweight="bold")
    ax_c.text(0.50, 0.72, "SCORE = {T, R, O}", ha="center", fontsize=PREF_FONT_PT, family="monospace")
    ax_c.text(0.50, 0.58, "VALUE = {V}", ha="center", fontsize=PREF_FONT_PT, family="monospace")
    ax_c.text(0.50, 0.44, "CONTEXT = {C}", ha="center", fontsize=PREF_FONT_PT, family="monospace")
    ax_c.text(0.88, 0.72, "Φ_SCORE = φ_T + φ_R + φ_O", ha="center", fontsize=PREF_FONT_PT, family="monospace")
    ax_c.text(0.88, 0.52, "Φ_SCORE + φ_V + φ_C = Δ_h", ha="center", fontsize=PREF_FONT_PT, family="monospace")
    ax_c.text(
        0.50,
        0.16,
        "Standard Owen value; 36 admissible block-contiguous permutations.",
        ha="center",
        fontsize=MIN_FONT_PT,
    )
    ax_c.text(0.50, 0.06, "Mixed cells are controlled counterfactuals.", ha="center", fontsize=MIN_FONT_PT, style="italic")

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
