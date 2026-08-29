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
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.34), gridspec_kw={"width_ratios": [1.2, 0.8, 1.0]})

    ax = axes[0]
    panel_label(ax, "(a)")
    data = np.array(matrix)
    im = ax.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(PHI_COLS)))
    ax.set_xticklabels([c.replace("phi_", "φ_") for c in PHI_COLS], fontsize=8)
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if abs(val) < 1e-12:
                txt = "0"
            else:
                txt = f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="black")
        ax.text(5.35, i, f"ΔU={delta_us[i]:+.3g}", va="center", fontsize=6.5)
    ax.set_xlim(-0.5, 5.8)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="normalized attribution")

    ax = axes[1]
    panel_label(ax, "(b)")
    owen = tc["owen"]
    labels = ["ΔU", "φ_T", "φ_R", "φ_O", "φ_V", "φ_C"]
    values = [float(tc["delta_u"]), float(owen["T"]), float(owen["R"]), float(owen["O"]), float(owen["V"]), float(owen["C"])]
    colors = [COLORS["endpoint"], COLORS["T"], COLORS["R"], COLORS["O"], COLORS["V"], COLORS["C"]]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, edgecolor="#333", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.text(0.5, -0.8, "50% / 50% Owen split (T × C)", ha="center", fontsize=7, transform=ax.transData)
    ax.set_xlabel("utility contribution", fontsize=8)

    ax = axes[2]
    panel_label(ax, "(c)")
    ax.axis("off")
    ax.text(0.0, 0.98, "Check", fontsize=8, fontweight="bold")
    ax.text(0.62, 0.98, "max error", fontsize=8, fontweight="bold", ha="right")
    y = 0.92
    for name, err in dashboard:
        ax.text(0.0, y, name, fontsize=6.8)
        ax.text(0.62, y, format_sci(err), ha="right", fontsize=6.8, family="monospace")
        y -= 0.055
    status = "overall PASS" if ef1["overall_pass"] else "overall FAIL"
    ax.text(0.0, 0.04, status, fontsize=9, fontweight="bold", color=COLORS["pass"] if ef1["overall_pass"] else COLORS["fail"])

    extracted = {
        "single_channel_delta_u": delta_us,
        "tc_interaction": tc,
        "dashboard": dashboard,
        "overall_pass": ef1["overall_pass"],
    }
    return save_figure(fig, "Fig3", script_path=Path(__file__), source_files=[EF1_PATH], extracted_values=extracted)


if __name__ == "__main__":
    render()
