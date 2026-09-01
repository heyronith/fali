"""Figure 3 — method validation (3 panels)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

_LEGACY = Path(__file__).resolve().parents[1] / "figure_scripts"
if str(_LEGACY) not in sys.path:
    sys.path.append(str(_LEGACY))

from figure_data import (  # noqa: E402
    EF1_PATH,
    ef1_grouped_validation_rows,
    load_ef1,
    single_channel_matrix,
)
from figure_style import (  # noqa: E402
    AXIS_LABEL_PT,
    CANDIDATE_DIR,
    COLORS,
    FIG_WIDTH_IN,
    MIN_FONT_PT,
    PHI_COLS,
    apply_style,
    fig_height_in,
    panel_heading,
    save_figure,
)


def render(out_dir: Path = CANDIDATE_DIR) -> dict:
    ef1 = load_ef1()
    row_names, matrix, delta_us = single_channel_matrix(ef1)
    tc = ef1["level_a"]["tc_interaction"]
    owen = tc["owen"]
    grouped = ef1_grouped_validation_rows(ef1)
    ep = ef1["level_b"]["independent_endpoints"]
    max_disc = float(ef1["level_a"].get("max_attribution_discrepancy", 1.11e-16))

    apply_style()
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height_in("Fig3")))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 0.95], width_ratios=[1.15, 0.85], hspace=0.40, wspace=0.35)
    gs_a = gs[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.22], wspace=0.05)
    ax_heat = fig.add_subplot(gs_a[0, 0])
    ax_du = fig.add_subplot(gs_a[0, 1])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    panel_heading(ax_heat, "(a)", "Single-factor attribution recovery")
    data = np.array(matrix)
    ax_heat.imshow(data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax_heat.set_xticks(range(len(PHI_COLS)))
    ax_heat.set_xticklabels([r"$\phi_T$", r"$\phi_R$", r"$\phi_O$", r"$\phi_V$", r"$\phi_C$"])
    ax_heat.set_yticks(range(len(row_names)))
    ax_heat.set_yticklabels([r.replace("-only", "") for r in row_names])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            txt = "0" if abs(val) < 1e-12 else f"{val:.2f}"
            ax_heat.text(j, i, txt, ha="center", va="center", fontsize=MIN_FONT_PT)
    ax_du.axis("off")
    ax_du.set_ylim(-0.5, len(row_names) - 0.5)
    ax_du.text(0, len(row_names) - 0.3, r"$\Delta U$", fontsize=MIN_FONT_PT, fontweight="bold")
    for i, du in enumerate(delta_us):
        ax_du.text(0, i, f"{du:+.6g}", va="center", fontsize=MIN_FONT_PT, family="monospace")
    ax_heat.text(0.5, -0.18, f"max discrepancy = {max_disc:.2e}", transform=ax_heat.transAxes,
                 ha="center", fontsize=MIN_FONT_PT)

    panel_heading(ax_b, "(b)", r"Pure $T \times C$ interaction")
    labels = [r"$\phi_T$", r"$\phi_R$", r"$\phi_O$", r"$\phi_V$", r"$\phi_C$"]
    values = [float(owen["T"]), float(owen["R"]), float(owen["O"]), float(owen["V"]), float(owen["C"])]
    colors = [COLORS["T"], COLORS["R"], COLORS["O"], COLORS["V"], COLORS["C"]]
    x = np.arange(len(labels))
    ax_b.bar(x, values, color=colors, edgecolor="#333", linewidth=0.5, width=0.65)
    ax_b.axhline(0, color="#333", linewidth=0.8)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels)
    ax_b.set_ylabel("attribution", fontsize=AXIS_LABEL_PT)
    ax_b.text(0.5, 0.95, r"$\Delta_h \approx 2$; expected 50/50 Owen split",
              transform=ax_b.transAxes, ha="center", fontsize=MIN_FONT_PT)

    panel_heading(ax_c, "(c)", "Transformer implementation invariants")
    ax_c.axis("off")
    ax_c.text(0.02, 0.92, "Check", fontsize=MIN_FONT_PT, fontweight="bold")
    ax_c.text(0.55, 0.92, "residual / status", fontsize=MIN_FONT_PT, fontweight="bold")
    y = 0.78
    for name, val in grouped:
        ax_c.text(0.02, y, name, fontsize=MIN_FONT_PT)
        ax_c.text(0.55, y, val, fontsize=MIN_FONT_PT, family="monospace")
        y -= 0.11
    inset = (
        f"intact effect {float(ep['intact_direct']):+.6f}\n"
        f"conditional effect {float(ep['conditional_direct']):+.6f}\n"
        f"gain {float(ep['conditional_gain_direct']):+.6f}"
    )
    ax_c.text(0.78, 0.55, inset, fontsize=MIN_FONT_PT, family="monospace", va="top",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#F8F8F8", edgecolor="#CCC", linewidth=0.5))

    extracted = {
        "single_channel_delta_u": delta_us,
        "tc_interaction": tc,
        "grouped_validation": grouped,
        "endpoints": {
            "intact": float(ep["intact_direct"]),
            "conditional": float(ep["conditional_direct"]),
            "gain": float(ep["conditional_gain_direct"]),
        },
        "overall_pass": ef1["overall_pass"],
    }
    return save_figure(
        fig, "Fig3", out_dir=out_dir, script_path=Path(__file__), source_files=[EF1_PATH],
        extracted_values=extracted,
    )


if __name__ == "__main__":
    render()
