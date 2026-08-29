"""Render Figure 1 — conditional importance to mechanism resolution."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from figure_style import COLORS, FIG_WIDTH_IN, apply_style, panel_label, save_figure  # noqa: E402


def render() -> dict:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.28))
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    panel_label(ax, "(a)")
    states = [
        ("State A — intact", "primary present", False),
        ("State B — perturbed context", "primary intervened", True),
    ]
    y0 = 0.72
    for i, (title, primary, perturbed) in enumerate(states):
        y = y0 - i * 0.38
        ax.text(0.02, y + 0.08, title, fontsize=9, fontweight="bold")
        box = FancyBboxPatch((0.02, y - 0.05), 0.42, 0.12, boxstyle="round,pad=0.01", linewidth=1, edgecolor="#333", facecolor="#F7F7F7")
        ax.add_patch(box)
        ax.text(0.05, y, "primary circuit", fontsize=7, color="#555")
        if perturbed:
            ax.text(0.18, y, "×", fontsize=14, color="#D55E00", fontweight="bold")
        ax.add_patch(FancyBboxPatch((0.28, y - 0.02), 0.12, 0.06, boxstyle="round,pad=0.01", linewidth=1.2, edgecolor=COLORS["T"], facecolor="#E6F2FA"))
        ax.text(0.29, y + 0.005, "head h", fontsize=7, color=COLORS["T"])
        eq = "E_A(h) = m(A, h present) − m(A, h zero)" if i == 0 else "E_B(h) = m(B, h present) − m(B, h zero)"
        ax.text(0.48, y + 0.01, eq, fontsize=7, family="monospace")

    ax = axes[1]
    panel_label(ax, "(b)")
    ax.text(0.5, 0.72, "Δ_h = E_B(h) − E_A(h)", ha="center", fontsize=11, fontweight="bold", family="monospace")
    ax.text(0.5, 0.48, "Detection: did importance change?", ha="center", fontsize=8)
    ax.text(0.5, 0.36, "FALI: which changes account for Δ_h?", ha="center", fontsize=8, fontweight="bold")
    arrow = FancyArrowPatch((0.72, 0.25), (0.95, 0.25), transform=ax.transAxes, arrowstyle="-|>", mutation_scale=12, linewidth=1.2)
    ax.add_patch(arrow)

    ax = axes[2]
    panel_label(ax, "(c)")
    tokens = [
        ("T", "target score(s)"),
        ("R", "rival scores"),
        ("O", "other valid scores"),
        ("V", "value content"),
        ("C", "surrounding context"),
    ]
    x = 0.04
    for sym, label in tokens:
        ax.add_patch(FancyBboxPatch((x, 0.58), 0.08, 0.12, boxstyle="round,pad=0.01", linewidth=1, edgecolor=COLORS[sym], facecolor="white"))
        ax.text(x + 0.04, 0.64, sym, ha="center", fontsize=10, fontweight="bold", color=COLORS[sym])
        ax.text(x + 0.04, 0.52, label, ha="center", fontsize=6.5, wrap=True)
        x += 0.19
    ax.text(0.5, 0.34, "φ_T + φ_R + φ_O + φ_V + φ_C = Δ_h", ha="center", fontsize=9, family="monospace")
    ax.text(0.5, 0.12, "Attributions are defined under an explicit counterfactual intervention game.", ha="center", fontsize=7, style="italic")

    return save_figure(
        fig,
        "Fig1",
        script_path=Path(__file__),
        source_files=[],
        extracted_values={"type": "deterministic_schematic"},
    )


if __name__ == "__main__":
    render()
