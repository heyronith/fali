"""Render Figure 4 — GPT-2 attention vs recruitment."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from figure_data import (  # noqa: E402
    EF4_CI_PATH,
    EF4_GATE_PATH,
    get_ef4_metric,
    load_ef4_ci,
    load_ef4_gate,
)
from figure_style import (  # noqa: E402
    COLORS,
    FIG_WIDTH_IN,
    MARKERS,
    MIN_FONT_PT,
    PREF_FONT_PT,
    apply_style,
    fig_height_in,
    panel_label,
    save_figure,
    zero_line,
)


def render() -> dict:
    df = load_ef4_ci()
    gate = load_ef4_gate()
    heads = ["L10H2", "L10H10"]
    head_markers = {"L10H2": "o", "L10H10": "^"}
    extracted = {"heads": {}}
    for head in heads:
        extracted["heads"][head] = {
            "delta_target_attn": get_ef4_metric(df, head, "delta_target_attn"),
            "conditional_gain": get_ef4_metric(df, head, "conditional_gain"),
            "phi_T": get_ef4_metric(df, head, "phi_T"),
            "phi_R": get_ef4_metric(df, head, "phi_R"),
            "Phi_SCORE": get_ef4_metric(df, head, "Phi_SCORE"),
            "phi_V": get_ef4_metric(df, head, "phi_V"),
            "phi_C": get_ef4_metric(df, head, "phi_C"),
        }

    apply_style()
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height_in("Fig4")))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 1.0], width_ratios=[1.0, 1.0], hspace=0.38, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    panel_label(ax_a, "(a)")
    for head in heads:
        d = extracted["heads"][head]
        x = d["delta_target_attn"]["point"]
        y = d["conditional_gain"]["point"]
        xerr = [[x - d["delta_target_attn"]["ci_low"]], [d["delta_target_attn"]["ci_high"] - x]]
        yerr = [[y - d["conditional_gain"]["ci_low"]], [d["conditional_gain"]["ci_high"] - y]]
        ax_a.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=head_markers[head],
            color=COLORS["endpoint"],
            markersize=7,
            capsize=3,
            linewidth=1.2,
            zorder=3,
            label=head,
        )
        ax_a.annotate(
            head,
            (x, y),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=PREF_FONT_PT,
            ha="left",
        )
    zero_line(ax_a, "x")
    zero_line(ax_a, "y")
    ax_a.set_xlabel("Change in target attention", fontsize=PREF_FONT_PT)
    ax_a.set_ylabel("Conditional gain", fontsize=PREF_FONT_PT)
    ax_a.margins(x=0.15, y=0.20)

    panel_label(ax_b, "(b)")
    rows = [
        ("L10H2  T", "L10H2", "phi_T", "T"),
        ("L10H2  R", "L10H2", "phi_R", "R"),
        ("L10H10 T", "L10H10", "phi_T", "T"),
        ("L10H10 R", "L10H10", "phi_R", "R"),
    ]
    y_positions = list(range(len(rows)))
    xmax = 0.0
    xmin = 0.0
    for i, (_label, head, metric, ch) in enumerate(rows):
        d = extracted["heads"][head][metric]
        ax_b.errorbar(
            d["point"],
            i,
            xerr=[[d["point"] - d["ci_low"]], [d["ci_high"] - d["point"]]],
            fmt=MARKERS[ch],
            color=COLORS[ch],
            capsize=3,
            markersize=7,
            zorder=3,
        )
        xmax = max(xmax, d["ci_high"])
        xmin = min(xmin, d["ci_low"])
    zero_line(ax_b, "x")
    ax_b.set_yticks(y_positions)
    ax_b.set_yticklabels([r[0] for r in rows], fontsize=PREF_FONT_PT)
    ax_b.set_xlim(min(-0.45, xmin - 0.08), xmax + 0.12)
    ax_b.set_xlabel("FALI attribution", fontsize=PREF_FONT_PT)
    ax_b.text(
        0.0,
        -0.22,
        f"EF-4 fine-score gate: {gate['EF4_FINE_SCORE_GATE']}",
        transform=ax_b.transAxes,
        fontsize=MIN_FONT_PT,
    )

    panel_label(ax_c, "(c)")
    metrics = [("Phi_SCORE", "SCORE"), ("phi_V", "V"), ("phi_C", "C"), ("conditional_gain", "gain")]
    y_positions = list(range(len(metrics)))
    xmax = 0.0
    xmin = 0.0
    for head in heads:
        for j, (metric, short) in enumerate(metrics):
            d = extracted["heads"][head][metric]
            y = j + (-0.18 if head == "L10H2" else 0.18)
            color = COLORS.get(short, COLORS["endpoint"])
            marker = head_markers[head]
            ax_c.errorbar(
                d["point"],
                y,
                xerr=[[d["point"] - d["ci_low"]], [d["ci_high"] - d["point"]]],
                fmt=marker,
                color=color,
                capsize=3,
                markersize=6,
                zorder=3,
                label=head if j == 0 else "_nolegend_",
            )
            xmax = max(xmax, d["ci_high"])
            xmin = min(xmin, d["ci_low"])
    zero_line(ax_c, "x")
    ax_c.set_yticks(y_positions)
    ax_c.set_yticklabels(["SCORE", "V", "C", "endpoint gain"], fontsize=PREF_FONT_PT)
    ax_c.set_xlim(min(-0.55, xmin - 0.08), xmax + 0.12)
    ax_c.set_xlabel("Coarse / endpoint contribution", fontsize=PREF_FONT_PT)
    ax_c.legend(
        loc="lower right",
        fontsize=MIN_FONT_PT,
        frameon=False,
        title="Head",
        title_fontsize=MIN_FONT_PT,
        markerscale=1.0,
    )

    return save_figure(
        fig,
        "Fig4",
        script_path=Path(__file__),
        source_files=[EF4_CI_PATH, EF4_GATE_PATH],
        extracted_values=extracted,
        tight=False,
        revision_note="visual/scientific-diagram correction only; no empirical recomputation",
    )


if __name__ == "__main__":
    render()
