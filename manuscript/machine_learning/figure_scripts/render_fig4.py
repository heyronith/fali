"""Render Figure 4 — GPT-2 attention vs recruitment."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from figure_data import (  # noqa: E402
    EF4_CI_PATH,
    EF4_GATE_PATH,
    get_ef4_metric,
    load_ef4_ci,
    load_ef4_gate,
)
from figure_style import COLORS, FIG_WIDTH_IN, MARKERS, apply_style, panel_label, save_figure, zero_line  # noqa: E402


def render() -> dict:
    df = load_ef4_ci()
    gate = load_ef4_gate()
    heads = ["L10H2", "L10H10"]
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
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.36), gridspec_kw={"width_ratios": [1.0, 1.05, 1.15]})
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.16, wspace=0.42)

    ax = axes[0]
    panel_label(ax, "(a)")
    callouts = {
        "L10H2": ("attention ↑, usefulness ↑", (12, 12)),
        "L10H10": ("attention ↑, usefulness ↓", (12, -18)),
    }
    for head in heads:
        d = extracted["heads"][head]
        x = d["delta_target_attn"]["point"]
        y = d["conditional_gain"]["point"]
        xerr = [[x - d["delta_target_attn"]["ci_low"]], [d["delta_target_attn"]["ci_high"] - x]]
        yerr = [[y - d["conditional_gain"]["ci_low"]], [d["conditional_gain"]["ci_high"] - y]]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", color=COLORS["endpoint"], markersize=5, capsize=3, linewidth=1, zorder=3)
        ax.annotate(
            head,
            (x, y),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            ha="left",
        )
        note, off = callouts[head]
        ax.annotate(note, (x, y), xytext=off, textcoords="offset points", fontsize=6.5, style="italic", ha="left")
    zero_line(ax, "x")
    zero_line(ax, "y")
    ax.set_xlabel("Change in target attention", fontsize=8)
    ax.set_ylabel("Conditional gain", fontsize=8)
    ax.margins(x=0.12, y=0.18)

    ax = axes[1]
    panel_label(ax, "(b)")
    rows = [
        ("L10H2 — T", "L10H2", "phi_T", "T"),
        ("L10H2 — R", "L10H2", "phi_R", "R"),
        ("L10H10 — T", "L10H10", "phi_T", "T"),
        ("L10H10 — R", "L10H10", "phi_R", "R"),
    ]
    xmax = 0.0
    for i, (label, head, metric, ch) in enumerate(rows):
        d = extracted["heads"][head][metric]
        ax.errorbar(d["point"], i, xerr=[[d["point"] - d["ci_low"]], [d["ci_high"] - d["point"]]], fmt=MARKERS[ch], color=COLORS[ch], capsize=3, markersize=6, zorder=3)
        xmax = max(xmax, d["ci_high"])
        ax.text(d["ci_high"] + 0.03, i, label, va="center", fontsize=7.5)
    zero_line(ax, "x")
    ax.set_yticks([])
    ax.set_xlim(min(-0.45, ax.get_xlim()[0]), xmax + 0.22)
    ax.set_xlabel("FALI attribution", fontsize=8)
    ax.text(0.02, -0.22, f"EF-4 fine-score gate: {gate['EF4_FINE_SCORE_GATE']}", transform=ax.transAxes, fontsize=7)

    ax = axes[2]
    panel_label(ax, "(c)")
    metrics = [("Phi_SCORE", "SCORE"), ("phi_V", "V"), ("phi_C", "C"), ("conditional_gain", "gain")]
    head_markers = {"L10H2": "o", "L10H10": "^"}
    offsets = {"L10H2": -0.15, "L10H10": 0.15}
    xmax = 0.0
    for head in heads:
        for j, (metric, short) in enumerate(metrics):
            d = extracted["heads"][head][metric]
            y = j + offsets[head]
            color = COLORS.get(short, COLORS["endpoint"])
            marker = head_markers[head] if metric == "conditional_gain" else MARKERS.get(short, "o")
            ax.errorbar(
                d["point"],
                y,
                xerr=[[d["point"] - d["ci_low"]], [d["ci_high"] - d["point"]]],
                fmt=marker,
                color=color,
                capsize=3,
                markersize=5,
                zorder=3,
            )
            xmax = max(xmax, d["ci_high"], abs(d["ci_low"]))
            if metric == "conditional_gain":
                ax.text(d["point"], y + 0.10, f"{head} endpoint", fontsize=6.5, ha="center")
            elif metric == "phi_C":
                ax.text(min(d["ci_high"] + 0.02, xmax + 0.05), y, f"{head} context", fontsize=6.5, va="center")
    zero_line(ax, "x")
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(["Φ_SCORE", "φ_V", "φ_C", "endpoint gain"], fontsize=8)
    ax.set_xlim(min(-0.55, ax.get_xlim()[0]), xmax + 0.18)
    ax.set_xlabel("Coarse / endpoint contribution", fontsize=8)

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
