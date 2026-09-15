"""Render Figure 3 — synthetic + implementation validation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from figure_data import (  # noqa: E402
    EF1_PATH,
    ef1_dashboard_rows,
    ef1_grouped_validation_rows,
    load_ef1,
    single_channel_matrix,
)
from figure_style import (  # noqa: E402
    COLORS,
    FIG_WIDTH_IN,
    MIN_FONT_PT,
    PHI_COLS,
    PREF_FONT_PT,
    apply_style,
    fig_height_in,
    panel_label,
    save_figure,
)


def render() -> dict:
    ef1 = load_ef1()
    row_names, matrix, delta_us = single_channel_matrix(ef1)
    tc = ef1["level_a"]["tc_interaction"]
    dashboard = ef1_dashboard_rows(ef1)
    grouped = ef1_grouped_validation_rows(ef1)

    apply_style()
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height_in("Fig3")))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.95], width_ratios=[0.58, 0.42], hspace=0.35, wspace=0.28)
    gs_top = gs[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.22], wspace=0.06)
    ax_heat = fig.add_subplot(gs_top[0, 0])
    ax_delta = fig.add_subplot(gs_top[0, 1])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    panel_label(ax_heat, "(a)")
    data = np.array(matrix)
    ax_heat.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax_heat.set_xticks(range(len(PHI_COLS)))
    ax_heat.set_xticklabels([c.replace("phi_", "φ_") for c in PHI_COLS], fontsize=PREF_FONT_PT)
    ax_heat.set_yticks(range(len(row_names)))
    ax_heat.set_yticklabels(row_names, fontsize=PREF_FONT_PT)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt = "0" if abs(val) < 1e-12 else f"{val:.2f}"
            ax_heat.text(j, i, txt, ha="center", va="center", fontsize=MIN_FONT_PT, color="black")

    ax_delta.set_xlim(0, 1)
    ax_delta.set_ylim(-0.5, len(row_names) - 0.5)
    ax_delta.axis("off")
    ax_delta.text(0.0, len(row_names) - 0.2, "ΔU", fontsize=PREF_FONT_PT, fontweight="bold")
    for i, du in enumerate(delta_us):
        ax_delta.text(0.0, i, f"{du:+.3g}", va="center", ha="left", fontsize=MIN_FONT_PT, family="monospace")

    panel_label(ax_b, "(b)")
    owen = tc["owen"]
    other = float(owen["R"]) + float(owen["O"]) + float(owen["V"])
    labels = ["total ΔU", "φ_T", "φ_C", "other channels"]
    values = [float(tc["delta_u"]), float(owen["T"]), float(owen["C"]), other]
    colors = [COLORS["endpoint"], COLORS["T"], COLORS["C"], COLORS["O"]]
    y = np.arange(len(labels))
    ax_b.barh(y, values, color=colors, edgecolor="#333", linewidth=0.5, height=0.55)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels, fontsize=PREF_FONT_PT)
    ax_b.axvline(0, color="black", linewidth=1.0)
    ax_b.set_xlabel("utility contribution", fontsize=PREF_FONT_PT)
    ax_b.text(
        0.5,
        -0.18,
        "T × C interaction → 50/50 Owen split",
        ha="center",
        fontsize=PREF_FONT_PT,
        transform=ax_b.transAxes,
    )

    panel_label(ax_c, "(c)")
    ax_c.axis("off")
    ax_c.text(0.0, 0.96, "Validation group", fontsize=PREF_FONT_PT, fontweight="bold")
    ax_c.text(0.88, 0.96, "max error / status", fontsize=PREF_FONT_PT, fontweight="bold", ha="right")
    y = 0.82
    for name, value in grouped:
        ax_c.text(0.0, y, name, fontsize=MIN_FONT_PT)
        ax_c.text(0.88, y, value, ha="right", fontsize=MIN_FONT_PT, family="monospace")
        y -= 0.11
    status = "Overall validation: PASS" if ef1["overall_pass"] else "Overall validation: FAIL"
    ax_c.text(
        0.0,
        0.06,
        status,
        fontsize=PREF_FONT_PT,
        fontweight="bold",
        color=COLORS["pass"] if ef1["overall_pass"] else COLORS["fail"],
    )

    extracted = {
        "single_channel_delta_u": delta_us,
        "tc_interaction": tc,
        "dashboard": dashboard,
        "overall_pass": ef1["overall_pass"],
    }
    return save_figure(
        fig,
        "Fig3",
        script_path=Path(__file__),
        source_files=[EF1_PATH],
        extracted_values=extracted,
        tight=False,
        revision_note="visual/scientific-diagram correction only; no empirical recomputation",
    )


if __name__ == "__main__":
    render()
