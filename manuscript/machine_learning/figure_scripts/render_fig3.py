"""Render Figure 3 — synthetic + implementation validation."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from figure_data import EF1_PATH, ef1_dashboard_rows, load_ef1, single_channel_matrix  # noqa: E402
from figure_style import (  # noqa: E402
    COLORS,
    FIG_WIDTH_IN,
    PHI_COLS,
    apply_style,
    format_sci,
    panel_label,
    save_figure,
)


def render() -> dict:
    ef1 = load_ef1()
    row_names, matrix, delta_us = single_channel_matrix(ef1)
    tc = ef1["level_a"]["tc_interaction"]
    dashboard = ef1_dashboard_rows(ef1)

    apply_style()
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.36))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.75, 1.0], wspace=0.45)
    gs_a = gs[0].subgridspec(1, 2, width_ratios=[1.0, 0.28], wspace=0.08)
    ax_heat = fig.add_subplot(gs_a[0, 0])
    ax_delta = fig.add_subplot(gs_a[0, 1])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2])

    panel_label(ax_heat, "(a)")
    data = np.array(matrix)
    im = ax_heat.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax_heat.set_xticks(range(len(PHI_COLS)))
    ax_heat.set_xticklabels([c.replace("phi_", "φ_") for c in PHI_COLS], fontsize=8)
    ax_heat.set_yticks(range(len(row_names)))
    ax_heat.set_yticklabels(row_names, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt = "0" if abs(val) < 1e-12 else f"{val:.2f}"
            ax_heat.text(j, i, txt, ha="center", va="center", fontsize=7, color="black")
    ax_delta.set_xlim(0, 1)
    ax_delta.set_ylim(-0.5, len(row_names) - 0.5)
    ax_delta.axis("off")
    for i, du in enumerate(delta_us):
        ax_delta.text(0.0, i, f"ΔU={du:+.3g}", va="center", ha="left", fontsize=6.5)
    cbar = fig.colorbar(im, ax=[ax_heat, ax_delta], fraction=0.035, pad=0.02, label="normalized attribution")

    panel_label(ax_b, "(b)")
    owen = tc["owen"]
    labels = ["ΔU", "φ_T", "φ_R", "φ_O", "φ_V", "φ_C"]
    values = [float(tc["delta_u"]), float(owen["T"]), float(owen["R"]), float(owen["O"]), float(owen["V"]), float(owen["C"])]
    colors = [COLORS["endpoint"], COLORS["T"], COLORS["R"], COLORS["O"], COLORS["V"], COLORS["C"]]
    y = np.arange(len(labels))
    ax_b.barh(y, values, color=colors, edgecolor="#333", linewidth=0.5)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels, fontsize=8)
    ax_b.axvline(0, color="black", linewidth=1.0)
    ax_b.text(0.5, 0.02, "50% / 50% Owen split (T × C)", ha="center", fontsize=7, transform=ax_b.transAxes)
    ax_b.set_xlabel("utility contribution", fontsize=8)

    panel_label(ax_c, "(c)")
    ax_c.axis("off")
    ax_c.text(0.0, 0.98, "Check", fontsize=8, fontweight="bold")
    ax_c.text(0.62, 0.98, "max error", fontsize=8, fontweight="bold", ha="right")
    y = 0.92
    for name, err in dashboard:
        ax_c.text(0.0, y, name, fontsize=6.6)
        ax_c.text(0.62, y, format_sci(err), ha="right", fontsize=6.6, family="monospace")
        y -= 0.052
    status = "overall PASS" if ef1["overall_pass"] else "overall FAIL"
    ax_c.text(0.0, 0.04, status, fontsize=9, fontweight="bold", color=COLORS["pass"] if ef1["overall_pass"] else COLORS["fail"])

    fig.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.12, wspace=0.35)

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
