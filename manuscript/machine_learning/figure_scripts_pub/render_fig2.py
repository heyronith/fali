"""Figure 2 — FALI intervention game (3 panels)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Rectangle

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from figure_style import (  # noqa: E402
    AXIS_LABEL_PT,
    CANDIDATE_DIR,
    COLORS,
    FIG_WIDTH_IN,
    MIN_FONT_PT,
    apply_style,
    fig_height_in,
    panel_heading,
    save_figure,
)
_LEGACY = Path(__file__).resolve().parents[1] / "figure_scripts"
if str(_LEGACY) not in sys.path:
    sys.path.append(str(_LEGACY))

from score_partition import (  # noqa: E402
    ILLUSTRATIVE_K_STAR,
    ILLUSTRATIVE_N,
    ILLUSTRATIVE_Q_STAR,
    build_score_partition,
    summarize_partition,
)


def _draw_partition(ax) -> dict:
    grid = build_score_partition()
    summary = summarize_partition(grid)
    n = ILLUSTRATIVE_N
    cell = 0.085
    x0, y0 = 0.08, 0.08
    for q in range(n):
        for k in range(n):
            x = x0 + k * cell
            y = y0 + (n - 1 - q) * cell
            player = grid[(q, k)]
            if player == "MASK":
                ax.add_patch(
                    Rectangle(
                        (x, y), cell, cell,
                        facecolor=COLORS["mask"], edgecolor="#999", hatch="///", linewidth=0.5,
                    )
                )
            else:
                ax.add_patch(Rectangle((x, y), cell, cell, facecolor="white", edgecolor=COLORS[player], linewidth=0.8))
                ax.text(x + cell / 2, y + cell / 2, player, ha="center", va="center",
                        fontsize=MIN_FONT_PT, color=COLORS[player], fontweight="bold")
    ax.text(x0 + ILLUSTRATIVE_K_STAR * cell + cell / 2, y0 - 0.02, r"$k^\ast$", ha="center", fontsize=MIN_FONT_PT)
    ax.text(x0 + n * cell + 0.02, y0 + (n - 1 - ILLUSTRATIVE_Q_STAR) * cell + cell / 2,
            r"$q^\ast$", va="center", fontsize=MIN_FONT_PT)
    ax.text(x0 - 0.03, y0 + n * cell / 2, "query $q$", rotation=90, va="center", ha="center", fontsize=MIN_FONT_PT)
    ax.text(x0 + n * cell / 2, y0 - 0.05, "key $k$", ha="center", fontsize=MIN_FONT_PT)
    defs = [
        (COLORS["T"], r"$T$: target $(q^\ast,k^\ast)$"),
        (COLORS["R"], r"$R$: rivals in $q^\ast$ row"),
        (COLORS["O"], r"$O$: other valid scores"),
        ("#999", r"$\times$: masked"),
    ]
    dx = 0.56
    for i, (c, txt) in enumerate(defs):
        ax.plot([dx, dx + 0.015], [0.22 + i * 0.10, 0.22 + i * 0.10], color=c, lw=2, solid_capstyle="butt")
        ax.text(dx + 0.02, 0.22 + i * 0.10, txt, va="center", fontsize=MIN_FONT_PT)
    ax.text(0.56, 0.92, "illustrative partition", fontsize=MIN_FONT_PT, style="italic", color="#666")
    return {
        "partition_counts": {"T": summary.n_T, "R": summary.n_R, "O": summary.n_O, "MASK": summary.n_mask},
        "final_row": summary.final_row,
    }


def render(out_dir: Path = CANDIDATE_DIR) -> dict:
    apply_style()
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height_in("Fig2")))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.05], width_ratios=[1.0, 1.0], hspace=0.42, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    for ax in (ax_a, ax_b, ax_c):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    panel_heading(ax_a, "(a)", "Score partition at the task-relevant query")
    partition_meta = _draw_partition(ax_a)

    panel_heading(ax_b, "(b)", "Hybrid candidate-head utility")
    ax_b.text(0.5, 0.90, r"$T\leftarrow B,\; R\leftarrow A,\; O\leftarrow B,\; V\leftarrow A,\; C\leftarrow B$",
              ha="center", fontsize=MIN_FONT_PT, family="serif")
    steps = [
        "hybrid score tensor",
        "causal masked softmax",
        "hybrid attention pattern $P$",
        r"$P\,V$",
        "candidate head output $z_h$",
        "inject full $z_h$ into context $C$",
    ]
    y = 0.78
    for i, step in enumerate(steps):
        ax_b.text(0.5, y, step, ha="center", fontsize=MIN_FONT_PT)
        if i < len(steps) - 1:
            ax_b.add_patch(FancyArrowPatch((0.5, y - 0.02), (0.5, y - 0.055), arrowstyle="-|>", mutation_scale=8, lw=0.8))
        y -= 0.11
    ax_b.text(0.5, 0.08, r"$U(\mathrm{cell}) = \mathcal{M}(z_h^{\mathrm{hybrid}}, C) - \mathcal{M}(z_h{=}0, C)$",
              ha="center", fontsize=MIN_FONT_PT, family="serif")

    panel_heading(ax_c, "(c)", "Owen allocation and coarse–fine reconciliation")
    ax_c.text(0.12, 0.72, r"$2^5 = 32$ hybrid states", ha="center", fontsize=AXIS_LABEL_PT)
    ax_c.text(0.12, 0.55, r"00000 $\rightarrow$ 11111", ha="center", fontsize=MIN_FONT_PT, family="monospace")
    mid = [
        r"$\mathrm{SCORE}=\{T,R,O\}$",
        r"$\mathrm{VALUE}=\{V\}$",
        r"$\mathrm{CONTEXT}=\{C\}$",
        "36 admissible block-contiguous permutations",
    ]
    for i, line in enumerate(mid):
        ax_c.text(0.50, 0.78 - i * 0.12, line, ha="center", fontsize=MIN_FONT_PT)
    right = [
        r"$\Delta_h = U(11111) - U(00000)$",
        r"$\phi_T+\phi_R+\phi_O+\phi_V+\phi_C=\Delta_h$",
        r"$\Phi_{\mathrm{SCORE}}=\phi_T+\phi_R+\phi_O$",
    ]
    for i, line in enumerate(right):
        ax_c.text(0.86, 0.72 - i * 0.14, line, ha="center", fontsize=MIN_FONT_PT, family="serif")
    ax_c.text(0.50, 0.08, "Attributions are defined by the explicit counterfactual game.",
              ha="center", fontsize=MIN_FONT_PT, style="italic", color="#555")

    return save_figure(
        fig, "Fig2", out_dir=out_dir, script_path=Path(__file__), source_files=[],
        extracted_values={"type": "schematic", **partition_meta}, provenance=False,
    )


if __name__ == "__main__":
    render()
