"""Render Figure 5 — GPT-2 vs Pythia mechanism reversal."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from figure_data import (  # noqa: E402
    EF4_CI_PATH,
    EF6_BOOT_PATH,
    EF6_EXACT_PATH,
    EF6_GATE_PATH,
    EF6_HYP_PATH,
    get_ef4_metric,
    get_ef6_metric,
    load_ef4_ci,
    load_ef6_boot,
    load_ef6_exact,
    load_ef6_gate,
)
from figure_style import COLORS, FIG_WIDTH_IN, MARKERS, apply_style, panel_label, save_figure, zero_line  # noqa: E402


def _forest(ax, data, title, signs) -> None:
    rows = [("φ_T", "phi_T", "T"), ("φ_R", "phi_R", "R")]
    for i, (label, key, ch) in enumerate(rows):
        d = data[key]
        ax.errorbar(
            d["point"],
            i,
            xerr=[[d["point"] - d["ci_low"]], [d["ci_high"] - d["point"]]],
            fmt=MARKERS[ch],
            color=COLORS[ch],
            capsize=3,
            markersize=6,
        )
        ax.text(d["ci_high"] + 0.02, i, label, va="center", fontsize=7.5)
    zero_line(ax, "x")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["T", "R"], fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.text(0.98, 0.92, signs[0], transform=ax.transAxes, ha="right", fontsize=8, color=COLORS["T"])
    ax.text(0.98, 0.78, signs[1], transform=ax.transAxes, ha="right", fontsize=8, color=COLORS["R"])


def render() -> dict:
    ef4 = load_ef4_ci()
    ef6_exact = load_ef6_exact()
    ef6_boot = load_ef6_boot()
    ef6_gate = load_ef6_gate()

    gpt2 = {
        "conditional_gain": get_ef4_metric(ef4, "L10H2", "conditional_gain"),
        "phi_T": get_ef4_metric(ef4, "L10H2", "phi_T"),
        "phi_R": get_ef4_metric(ef4, "L10H2", "phi_R"),
    }
    pythia = {
        "conditional_gain": get_ef6_metric(ef6_exact, ef6_boot, "conditional_gain"),
        "phi_T": get_ef6_metric(ef6_exact, ef6_boot, "phi_T"),
        "phi_R": get_ef6_metric(ef6_exact, ef6_boot, "phi_R"),
    }
    extracted = {"gpt2_l10h2": gpt2, "pythia_l8h2": pythia, "ef6_gates": ef6_gate["gates"], "ef6_gate": ef6_gate["EF6_GATE"]}

    apply_style()
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.46))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.1, 0.55], hspace=0.55, wspace=0.28)
    ax_gpt_gain = fig.add_subplot(gs[0, 0])
    ax_pyt_gain = fig.add_subplot(gs[0, 1])
    ax_gpt_forest = fig.add_subplot(gs[1, 0])
    ax_pyt_forest = fig.add_subplot(gs[1, 1])
    ax_gate = fig.add_subplot(gs[2, :])

    panel_label(ax_gpt_gain, "(a)")
    for ax, data, title, ylab in (
        (ax_gpt_gain, gpt2["conditional_gain"], "GPT-2 IOI / L10H2", "Conditional gain\n(IO−S logit units)"),
        (ax_pyt_gain, pythia["conditional_gain"], "Pythia induction / L8H2", "Conditional gain\n(target log-prob units)"),
    ):
        d = data
        ax.errorbar(
            [0],
            [d["point"]],
            yerr=[[d["point"] - d["ci_low"]], [d["ci_high"] - d["point"]]],
            fmt=MARKERS["endpoint"],
            color=COLORS["endpoint"],
            capsize=4,
            markersize=7,
        )
        zero_line(ax, "y")
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_title(title, fontsize=8)
        ax.set_ylabel(ylab, fontsize=7.5)
    fig.text(0.5, 0.97, "positive conditional recruitment", ha="center", fontsize=8, fontweight="bold")

    panel_label(ax_gpt_forest, "(b)")
    _forest(ax_gpt_forest, gpt2, "GPT-2 IOI", ("T +", "R −"))
    _forest(ax_pyt_forest, pythia, "Pythia induction", ("T −", "R +"))
    fig.text(0.5, 0.56, "Same recruitment label; opposite fine-score signs", ha="center", fontsize=8, style="italic")

    panel_label(ax_gate, "(c)")
    ax_gate.axis("off")
    gates = ef6_gate["gates"]
    labels = [
        ("G0 behavior", gates["G0"]),
        ("G1 primary causal importance", gates["G1"]),
        ("G2 conditional recruitment", gates["G2"]),
        ("G3 predicted φ_T > 0", gates["G3"]),
        ("G4 predicted φ_R < 0", gates["G4"]),
    ]
    x = 0.02
    for name, status in labels:
        mark = "confirmed" if status == "CONFIRMED" else "NOT CONFIRMED"
        sym = "✓" if status == "CONFIRMED" else "✗"
        ax_gate.text(x, 0.55, f"{name} {sym}", fontsize=7.5)
        x += 0.19
    ax_gate.text(0.02, 0.15, f"EF-6 transfer gate: {ef6_gate['EF6_GATE']}", fontsize=9, fontweight="bold")
    ax_gate.text(0.42, 0.15, "failure caused by sign reversal, not absence of recruitment", fontsize=7.5, style="italic")

    return save_figure(
        fig,
        "Fig5",
        script_path=Path(__file__),
        source_files=[EF4_CI_PATH, EF6_EXACT_PATH, EF6_BOOT_PATH, EF6_GATE_PATH, EF6_HYP_PATH],
        extracted_values=extracted,
    )


if __name__ == "__main__":
    render()
