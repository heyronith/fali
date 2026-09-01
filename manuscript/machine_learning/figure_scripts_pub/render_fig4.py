"""Figure 4 — GPT-2 IOI main result (3 panels)."""

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
    EF4_CI_PATH,
    EF4_GATE_PATH,
    get_ef4_metric,
    load_ef4_ci,
    load_ef4_gate,
)
from figure_style import (  # noqa: E402
    AXIS_LABEL_PT,
    CANDIDATE_DIR,
    COLORS,
    FIG_WIDTH_IN,
    MARKERS,
    MIN_FONT_PT,
    apply_style,
    fig_height_in,
    panel_heading,
    save_figure,
    zero_line,
)

HEADS = ["L10H2", "L10H10"]
CHANNELS = ["phi_T", "phi_R", "phi_O", "phi_V", "phi_C"]
CH_SHORT = ["T", "R", "O", "V", "C"]


def _waterfall(ax, head: str, data: dict, x_offset: float) -> None:
    score = data["Phi_SCORE"]["point"]
    val = data["phi_V"]["point"]
    ctx = data["phi_C"]["point"]
    gain = data["conditional_gain"]["point"]
    levels = [0, score, score + val, score + val + ctx, gain]
    labels = ["0", "SCORE", "VALUE", "CONTEXT", "gain"]
    xs = [x_offset + i * 0.18 for i in range(5)]
    for i in range(4):
        dy = levels[i + 1] - levels[i]
        color = COLORS["endpoint"] if labels[i + 1] == "gain" else COLORS.get(labels[i + 1][0] if labels[i + 1] != "gain" else "C", "#333")
        if labels[i + 1] == "SCORE":
            color = COLORS["T"]
        elif labels[i + 1] == "VALUE":
            color = COLORS["V"]
        elif labels[i + 1] == "CONTEXT":
            color = COLORS["C"]
        ax.bar(xs[i + 1], dy, bottom=levels[i], width=0.14, color=color, edgecolor="#333", linewidth=0.5)
        if i < 3:
            ax.plot([xs[i + 1] + 0.07, xs[i + 2] - 0.07], [levels[i + 1], levels[i + 1]], color="#666", lw=0.7)
    ax.text(x_offset + 0.36, 0.92, head, transform=ax.transAxes, ha="center", fontsize=MIN_FONT_PT, fontweight="bold")
    zero_line(ax, "y")


def render(out_dir: Path = CANDIDATE_DIR) -> dict:
    df = load_ef4_ci()
    gate = load_ef4_gate()
    extracted = {"heads": {}, "gate": gate["EF4_FINE_SCORE_GATE"]}
    for head in HEADS:
        extracted["heads"][head] = {
            m: get_ef4_metric(df, head, m)
            for m in ["delta_target_attn", "conditional_gain", "phi_T", "phi_R", "phi_O", "phi_V", "phi_C", "Phi_SCORE"]
        }

    apply_style()
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height_in("Fig4")))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.15], width_ratios=[1.0, 1.0], hspace=0.42, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_ov = fig.add_axes([0.54, 0.10, 0.12, 0.20])  # O/V inset
    ax_c = fig.add_subplot(gs[1, 1])

    panel_heading(ax_a, "(a)", "Target-attention change versus conditional gain")
    markers = {"L10H2": "o", "L10H10": "^"}
    for head in HEADS:
        d = extracted["heads"][head]
        x, y = d["delta_target_attn"]["point"], d["conditional_gain"]["point"]
        xerr = [[x - d["delta_target_attn"]["ci_low"]], [d["delta_target_attn"]["ci_high"] - x]]
        yerr = [[y - d["conditional_gain"]["ci_low"]], [d["conditional_gain"]["ci_high"] - y]]
        ax_a.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=markers[head], color=COLORS["endpoint"],
                      markersize=6, capsize=3, linewidth=1, zorder=3)
        ax_a.annotate(head, (x, y), xytext=(8, 6), textcoords="offset points", fontsize=MIN_FONT_PT)
    zero_line(ax_a, "x")
    zero_line(ax_a, "y")
    ax_a.set_xlabel(r"$\Delta$ target attention", fontsize=AXIS_LABEL_PT)
    ax_a.set_ylabel("conditional gain", fontsize=AXIS_LABEL_PT)

    panel_heading(ax_b, "(b)", "Fine FALI attribution by candidate head")
    rows = []
    for head in HEADS:
        for ch, metric in zip(CH_SHORT, CHANNELS):
            rows.append((f"{head}  {ch}", head, metric, ch))
    ypos = np.arange(len(rows))
    for i, (_, head, metric, ch) in enumerate(rows):
        d = extracted["heads"][head][metric]
        if ch in ("O", "V"):
            continue  # plotted in inset
        ax_b.errorbar(d["point"], i, xerr=[[d["point"] - d["ci_low"]], [d["ci_high"] - d["point"]]],
                      fmt=MARKERS[ch], color=COLORS[ch], capsize=2, markersize=5, zorder=3)
    ax_b.set_yticks([0, 1, 2, 5, 6, 7])
    ax_b.set_yticklabels([rows[i][0] for i in [0, 1, 2, 3, 4, 5]])
    zero_line(ax_b, "x")
    ax_b.set_xlabel("FALI attribution", fontsize=AXIS_LABEL_PT)
    ax_b.text(0.02, -0.14, "main scale: T, R, C", transform=ax_b.transAxes, fontsize=MIN_FONT_PT, style="italic")

    # O/V inset
    ax_ov.set_title(r"$O/V$ (zero scale)", fontsize=MIN_FONT_PT, pad=2)
    inset_rows = [(h, m, c) for h in HEADS for m, c in zip(["phi_O", "phi_V"], ["O", "V"])]
    for i, (head, metric, ch) in enumerate(inset_rows):
        d = extracted["heads"][head][metric]
        ax_ov.errorbar(d["point"], i, xerr=[[d["point"] - d["ci_low"]], [d["ci_high"] - d["point"]]],
                       fmt=MARKERS[ch], color=COLORS[ch], capsize=2, markersize=4)
    ax_ov.set_yticks(range(len(inset_rows)))
    ax_ov.set_yticklabels([f"{h} {c}" for h, _, c in inset_rows], fontsize=MIN_FONT_PT)
    zero_line(ax_ov, "x")
    ax_ov.tick_params(labelsize=MIN_FONT_PT)

    panel_heading(ax_c, "(c)", "SCORE, VALUE, and CONTEXT contributions")
    ax_c.set_xlim(-0.05, 1.0)
    ax_c.set_ylabel("cumulative conditional gain", fontsize=AXIS_LABEL_PT)
    _waterfall(ax_c, "L10H2", extracted["heads"]["L10H2"], 0.08)
    _waterfall(ax_c, "L10H10", extracted["heads"]["L10H10"], 0.55)
    ax_c.text(0.5, -0.12, "waterfall: 0 → SCORE → VALUE → CONTEXT → gain",
              transform=ax_c.transAxes, ha="center", fontsize=MIN_FONT_PT, style="italic")

    return save_figure(
        fig, "Fig4", out_dir=out_dir, script_path=Path(__file__),
        source_files=[EF4_CI_PATH, EF4_GATE_PATH], extracted_values=extracted,
    )


if __name__ == "__main__":
    render()
