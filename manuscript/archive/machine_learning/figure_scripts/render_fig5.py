"""Render Figure 5 — GPT-2 vs Pythia mechanism reversal."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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
from figure_style import (  # noqa: E402
    COLORS,
    FIG_WIDTH_IN,
    MARKERS,
    MIN_FONT_PT,
    PANEL_LABEL_PT,
    PREF_FONT_PT,
    apply_style,
    fig_height_in,
    panel_label,
    save_figure,
    zero_line,
)

GATE_ROWS = [
    ("G0", "Behavior", "G0"),
    ("G1", "Primary causal importance", "G1"),
    ("G2", "Conditional recruitment", "G2"),
    ("G3", "Predicted phi_T > 0", "G3"),
    ("G4", "Predicted phi_R < 0", "G4"),
]


def _forest(ax, data, title) -> None:
    rows = [("φ_T", "phi_T", "T"), ("φ_R", "phi_R", "R")]
    for i, (label, key, ch) in enumerate(rows):
        d = data[key]
        ax.errorbar(
            d["point"],
            i,
            xerr=[[d["point"] - d["ci_low"]], [d["ci_high"] - d["point"]]],
            fmt=MARKERS[ch],
            color=COLORS[ch],
            capsize=4,
            markersize=8,
            zorder=3,
        )
    zero_line(ax, "x")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["φ_T", "φ_R"], fontsize=PREF_FONT_PT)
    ax.set_title(title, fontsize=PREF_FONT_PT, pad=8)
    ax.set_xlabel("FALI attribution", fontsize=PREF_FONT_PT)


def _draw_gate_table(ax, ef6_gate: dict) -> None:
    ax.axis("off")
    gates = ef6_gate["gates"]
    table_data = [[gate, criterion, gates[key]] for gate, criterion, key in GATE_ROWS]
    table = ax.table(
        cellText=table_data,
        colLabels=["Gate", "Preregistered criterion", "Outcome"],
        cellLoc="left",
        colLoc="left",
        loc="upper center",
        bbox=[0.0, 0.22, 1.0, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(PREF_FONT_PT)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        cell.set_edgecolor("#666666")
        cell.set_height(0.14)
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#F5F5F5")
        elif col == 2:
            outcome = cell.get_text().get_text()
            cell.set_facecolor("#F7F7F7")
            cell.get_text().set_text(outcome.replace("_", " "))
    ax.text(0.0, 0.10, f"EF-6 transfer gate: {ef6_gate['EF6_GATE']}", fontsize=PREF_FONT_PT, fontweight="bold", color=COLORS["fail"])
    ax.text(
        0.0,
        0.02,
        "Mechanism-transfer failure; recruitment remains confirmed.",
        fontsize=MIN_FONT_PT,
        style="italic",
        color=COLORS["neutral"],
    )


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
    extracted = {
        "gpt2_l10h2": gpt2,
        "pythia_l8h2": pythia,
        "ef6_gates": ef6_gate["gates"],
        "ef6_gate": ef6_gate["EF6_GATE"],
    }

    apply_style()
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height_in("Fig5")))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.45, 1.15], hspace=0.55, wspace=0.32)
    ax_gpt_gain = fig.add_subplot(gs[0, 0])
    ax_pyt_gain = fig.add_subplot(gs[0, 1])
    ax_gpt_forest = fig.add_subplot(gs[1, 0])
    ax_pyt_forest = fig.add_subplot(gs[1, 1])
    ax_gate = fig.add_subplot(gs[2, :])
    fig.subplots_adjust(left=0.12, right=0.96, top=0.94, bottom=0.06)
    fig.text(0.02, 0.92, "(a)", fontsize=PANEL_LABEL_PT, fontweight="bold", ha="left", va="top")
    for ax, data, title, ylab in (
        (ax_gpt_gain, gpt2["conditional_gain"], "GPT-2 IOI — L10H2", "Conditional gain\n(IO−S logit units)"),
        (ax_pyt_gain, pythia["conditional_gain"], "Pythia induction — L8H2", "Conditional gain\n(target log-prob units)"),
    ):
        d = data
        ax.errorbar(
            [0],
            [d["point"]],
            yerr=[[d["point"] - d["ci_low"]], [d["ci_high"] - d["point"]]],
            fmt=MARKERS["endpoint"],
            color=COLORS["endpoint"],
            capsize=4,
            markersize=8,
            zorder=3,
        )
        zero_line(ax, "y")
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_title(title, fontsize=PREF_FONT_PT)
        ax.set_ylabel(ylab, fontsize=PREF_FONT_PT)
        ax.tick_params(labelsize=MIN_FONT_PT)
        ax.margins(y=0.28)
    fig.text(
        0.5,
        0.66,
        "Positive conditional gain in both studied settings",
        ha="center",
        fontsize=PREF_FONT_PT,
        fontweight="bold",
    )

    panel_label(ax_gpt_forest, "(b)")
    _forest(ax_gpt_forest, gpt2, "GPT-2 IOI")
    _forest(ax_pyt_forest, pythia, "Pythia induction")
    fig.text(
        0.5,
        0.36,
        "Opposite target/rival signs",
        ha="center",
        fontsize=PREF_FONT_PT,
        style="italic",
    )

    panel_label(ax_gate, "(c)")
    _draw_gate_table(ax_gate, ef6_gate)

    return save_figure(
        fig,
        "Fig5",
        script_path=Path(__file__),
        source_files=[EF4_CI_PATH, EF6_EXACT_PATH, EF6_BOOT_PATH, EF6_GATE_PATH, EF6_HYP_PATH],
        extracted_values=extracted,
        tight=False,
        revision_note="visual/scientific-diagram correction only; no empirical recomputation",
    )


if __name__ == "__main__":
    render()
