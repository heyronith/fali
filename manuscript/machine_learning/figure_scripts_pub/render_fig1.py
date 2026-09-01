"""Figure 1 — what FALI explains (no panels)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    save_figure,
)


def render(out_dir: Path = CANDIDATE_DIR) -> dict:
    apply_style()
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, fig_height_in("Fig1")))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # State A (left)
    ax.text(0.12, 0.82, "State A: intact context", ha="center", fontsize=AXIS_LABEL_PT, fontweight="bold")
    ax.text(0.12, 0.70, "candidate head h", ha="center", fontsize=MIN_FONT_PT)
    ax.add_patch(FancyArrowPatch((0.12, 0.66), (0.12, 0.58), arrowstyle="-|>", mutation_scale=10, lw=0.9))
    ax.text(0.12, 0.52, "output-grounded causal usefulness", ha="center", fontsize=MIN_FONT_PT)
    ax.text(0.12, 0.46, r"$E_A(h)$", ha="center", fontsize=AXIS_LABEL_PT, family="serif")

    # State B (right)
    ax.text(0.88, 0.82, "State B: perturbed context", ha="center", fontsize=AXIS_LABEL_PT, fontweight="bold")
    ax.text(0.88, 0.70, "candidate head h", ha="center", fontsize=MIN_FONT_PT)
    ax.add_patch(FancyArrowPatch((0.88, 0.66), (0.88, 0.58), arrowstyle="-|>", mutation_scale=10, lw=0.9))
    ax.text(0.88, 0.52, "output-grounded causal usefulness", ha="center", fontsize=MIN_FONT_PT)
    ax.text(0.88, 0.46, r"$E_B(h)$", ha="center", fontsize=AXIS_LABEL_PT, family="serif")

    # Center perturbation arrow
    ax.add_patch(
        FancyArrowPatch(
            (0.28, 0.72),
            (0.72, 0.72),
            arrowstyle="-|>",
            mutation_scale=12,
            lw=1.0,
            color="#333333",
        )
    )
    ax.text(0.50, 0.76, "primary-circuit perturbation", ha="center", fontsize=MIN_FONT_PT)

    # Delta equation
    ax.text(0.50, 0.30, r"$\Delta_h = E_B(h) - E_A(h)$", ha="center", fontsize=AXIS_LABEL_PT + 0.5, family="serif")
    ax.text(0.50, 0.22, "change in candidate-head causal usefulness", ha="center", fontsize=MIN_FONT_PT, style="italic")

    # Factor decomposition line
    ax.text(0.50, 0.12, r"$\Delta_h \longrightarrow \{T,\; R,\; O,\; V,\; C\}$", ha="center", fontsize=AXIS_LABEL_PT, family="serif")
    labels = [
        ("T", "target score(s)"),
        ("R", "rival scores"),
        ("O", "other valid scores"),
        ("V", "value vectors/content"),
        ("C", "surrounding-network context"),
    ]
    xs = [0.08, 0.26, 0.44, 0.62, 0.80]
    for x, (sym, lab) in zip(xs, labels):
        ax.text(x, 0.05, sym, ha="center", fontsize=MIN_FONT_PT, fontweight="bold", color=COLORS[sym])
        ax.text(x, 0.01, lab, ha="center", fontsize=MIN_FONT_PT, color=COLORS["neutral"])

    return save_figure(
        fig,
        "Fig1",
        out_dir=out_dir,
        script_path=Path(__file__),
        source_files=[],
        extracted_values={"type": "schematic_no_empirical_data"},
        provenance=False,
    )


if __name__ == "__main__":
    render()
