"""Render Figure 1 — conditional importance to mechanism resolution."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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


def render() -> dict:
    apply_style()
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height_in("Fig1")))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.05], width_ratios=[0.58, 0.42], hspace=0.28, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    for ax in (ax_a, ax_b, ax_c):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    panel_label(ax_a, "(a)")
    states = [
        ("State A — intact", False),
        ("State B — perturbed context", True),
    ]
    y0 = 0.78
    for i, (title, perturbed) in enumerate(states):
        y = y0 - i * 0.42
        ax_a.text(0.03, y + 0.10, title, fontsize=PREF_FONT_PT, fontweight="bold")
        ax_a.add_patch(
            FancyBboxPatch(
                (0.03, y - 0.04),
                0.38,
                0.14,
                boxstyle="round,pad=0.012",
                linewidth=1,
                edgecolor="#333",
                facecolor="#F7F7F7",
            )
        )
        ax_a.text(0.06, y + 0.02, "primary circuit", fontsize=MIN_FONT_PT, color="#555")
        if perturbed:
            ax_a.text(0.20, y + 0.02, "×", fontsize=14, color="#D55E00", fontweight="bold")
        ax_a.add_patch(
            FancyBboxPatch(
                (0.24, y),
                0.14,
                0.08,
                boxstyle="round,pad=0.012",
                linewidth=1.2,
                edgecolor=COLORS["T"],
                facecolor="#E6F2FA",
            )
        )
        ax_a.text(0.31, y + 0.03, "head h", ha="center", fontsize=MIN_FONT_PT, color=COLORS["T"])
        endpoint = "Candidate usefulness: E_A(h)" if i == 0 else "Candidate usefulness: E_B(h)"
        ax_a.text(0.52, y + 0.04, endpoint, fontsize=PREF_FONT_PT, va="center")

    panel_label(ax_b, "(b)")
    ax_b.text(0.5, 0.74, "Δ_h = E_B(h) − E_A(h)", ha="center", fontsize=11, fontweight="bold", family="monospace")
    ax_b.text(0.5, 0.50, "Detection: did usefulness change?", ha="center", fontsize=PREF_FONT_PT)
    ax_b.text(0.5, 0.36, "FALI: what accounts for the change?", ha="center", fontsize=PREF_FONT_PT, fontweight="bold")
    ax_b.add_patch(
        FancyArrowPatch((0.72, 0.22), (0.94, 0.22), transform=ax_b.transAxes, arrowstyle="-|>", mutation_scale=12, linewidth=1.2)
    )

    panel_label(ax_c, "(c)")
    tokens = [
        ("T", "target score(s)"),
        ("R", "rival scores"),
        ("O", "other valid scores"),
        ("V", "value content"),
        ("C", "surrounding context"),
    ]
    xs = [0.04, 0.22, 0.40, 0.58, 0.76]
    for x, (sym, label) in zip(xs, tokens):
        ax_c.add_patch(
            FancyBboxPatch(
                (x, 0.58),
                0.16,
                0.16,
                boxstyle="round,pad=0.012",
                linewidth=1,
                edgecolor=COLORS[sym],
                facecolor="white",
            )
        )
        ax_c.text(x + 0.08, 0.66, sym, ha="center", fontsize=10, fontweight="bold", color=COLORS[sym])
        ax_c.text(x + 0.08, 0.52, label, ha="center", fontsize=MIN_FONT_PT)
    ax_c.text(0.5, 0.30, "φ_T + φ_R + φ_O + φ_V + φ_C = Δ_h", ha="center", fontsize=PREF_FONT_PT, family="monospace")
    ax_c.text(
        0.5,
        0.10,
        "Controlled counterfactual accounting; not five unique natural pathways.",
        ha="center",
        fontsize=MIN_FONT_PT,
        style="italic",
    )

    return save_figure(
        fig,
        "Fig1",
        script_path=Path(__file__),
        source_files=[],
        extracted_values={"type": "deterministic_schematic"},
        tight=False,
    )


if __name__ == "__main__":
    render()
