"""Figure 5 — cross-setting transfer (2 panels)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

_LEGACY = Path(__file__).resolve().parents[1] / "figure_scripts"
if str(_LEGACY) not in sys.path:
    sys.path.append(str(_LEGACY))

from figure_data import (  # noqa: E402
    EF4_CI_PATH,
    EF6_BOOT_PATH,
    EF6_EXACT_PATH,
    EF6_GATE_PATH,
    get_ef4_metric,
    get_ef6_metric,
    load_ef4_ci,
    load_ef6_boot,
    load_ef6_exact,
    load_ef6_gate,
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


def _gain_panel(ax, data: dict, title: str, ylab: str) -> None:
    d = data
    ax.errorbar([0], [d["point"]], yerr=[[d["point"] - d["ci_low"]], [d["ci_high"] - d["point"]]],
                fmt=MARKERS["endpoint"], color=COLORS["endpoint"], capsize=4, markersize=7, zorder=3)
    zero_line(ax, "y")
    ax.set_xlim(-0.4, 0.4)
    ax.set_xticks([])
    ax.set_title(title, fontsize=AXIS_LABEL_PT, pad=6)
    ax.set_ylabel(ylab, fontsize=MIN_FONT_PT)
    ax.tick_params(labelsize=MIN_FONT_PT)


def _trc_forest(ax, data: dict, title: str) -> None:
    rows = [("T", "phi_T"), ("R", "phi_R"), ("C", "phi_C")]
    for i, (ch, key) in enumerate(rows):
        d = data[key]
        alpha = 0.55 if ch == "C" else 1.0
        ax.errorbar(d["point"], i, xerr=[[d["point"] - d["ci_low"]], [d["ci_high"] - d["point"]]],
                    fmt=MARKERS[ch], color=COLORS[ch], capsize=3, markersize=6, zorder=3, alpha=alpha)
    zero_line(ax, "x")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels([r"$\phi_T$", r"$\phi_R$", r"$\phi_C$"], fontsize=MIN_FONT_PT)
    ax.set_title(title, fontsize=AXIS_LABEL_PT, pad=6)
    ax.set_xlabel("FALI attribution", fontsize=MIN_FONT_PT)
    ax.tick_params(labelsize=MIN_FONT_PT)


def render(out_dir: Path = CANDIDATE_DIR) -> dict:
    ef4 = load_ef4_ci()
    ef6_exact = load_ef6_exact()
    ef6_boot = load_ef6_boot()
    ef6_gate = load_ef6_gate()

    gpt2 = {
        "conditional_gain": get_ef4_metric(ef4, "L10H2", "conditional_gain"),
        "phi_T": get_ef4_metric(ef4, "L10H2", "phi_T"),
        "phi_R": get_ef4_metric(ef4, "L10H2", "phi_R"),
        "phi_C": get_ef4_metric(ef4, "L10H2", "phi_C"),
    }
    pythia = {
        "conditional_gain": get_ef6_metric(ef6_exact, ef6_boot, "conditional_gain"),
        "phi_T": get_ef6_metric(ef6_exact, ef6_boot, "phi_T"),
        "phi_R": get_ef6_metric(ef6_exact, ef6_boot, "phi_R"),
        "phi_C": get_ef6_metric(ef6_exact, ef6_boot, "phi_C"),
    }
    extracted = {
        "gpt2_l10h2": gpt2,
        "pythia_l8h2": pythia,
        "ef6_gate": ef6_gate["EF6_GATE"],
        "ef6_gates": ef6_gate["gates"],
    }

    apply_style()
    fig = plt.figure(figsize=(FIG_WIDTH_IN, fig_height_in("Fig5")))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.2], hspace=0.45, wspace=0.35)
    ax_g1 = fig.add_subplot(gs[0, 0])
    ax_g2 = fig.add_subplot(gs[0, 1])
    ax_f1 = fig.add_subplot(gs[1, 0])
    ax_f2 = fig.add_subplot(gs[1, 1])

    panel_heading(ax_g1, "(a)", "Positive conditional recruitment in both studied settings")
    _gain_panel(ax_g1, gpt2["conditional_gain"], "GPT-2 IOI — L10H2", "conditional gain\n(IO−S logit units)")
    _gain_panel(ax_g2, pythia["conditional_gain"], "Pythia-160M induction — L8H2", "conditional gain\n(target log-prob units)")
    fig.text(0.5, 0.52, "separate task-specific scales (not magnitude-comparable)", ha="center",
             fontsize=MIN_FONT_PT, style="italic")

    panel_heading(ax_f1, "(b)", "Target and rival attribution signs reverse")
    _trc_forest(ax_f1, gpt2, "GPT-2 IOI — L10H2")
    _trc_forest(ax_f2, pythia, "Pythia induction — L8H2")

    return save_figure(
        fig, "Fig5", out_dir=out_dir, script_path=Path(__file__),
        source_files=[EF4_CI_PATH, EF6_EXACT_PATH, EF6_BOOT_PATH, EF6_GATE_PATH],
        extracted_values=extracted,
    )


if __name__ == "__main__":
    render()
