"""
generate_new_figures.py

Generates two publication-quality figures for the FALI paper:
  1. Sequential Isolation Bar Chart (Conditions α, β, γ) — retires the
     intra-layer parallel communication hypothesis.
  2. QK Cosine Similarity Shift Lollipop Chart — geometric evidence of
     active vector realignment across all Layer 10 heads.

Data sourced from fali_rigorous_validation.py run outputs.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Global Style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

# ── Color Palette ─────────────────────────────────────────────────────
# A cohesive, publication-grade palette
COL_CLEAN    = "#4C72B0"   # Muted blue
COL_ALPHA    = "#DD8452"   # Warm orange
COL_BETA     = "#55A868"   # Green
COL_GAMMA    = "#C44E52"   # Red
COL_LN_FROZEN = "#8172B3"  # Purple
COL_POS      = "#2ca02c"   # Positive shift green
COL_NEG      = "#d62728"   # Negative shift red
COL_HIGHLIGHT = "#ff7f0e"  # Accent for L10H10


# ======================================================================
# FIGURE 1: Sequential Isolation Grouped Bar Chart
# ======================================================================
def generate_isolation_bar_chart():
    """
    Grouped bar chart showing L10H10's attention to IO under five conditions:
      Clean, Condition α (L9H9 only), Condition β (L10H0 only),
      Condition γ (Co-ablation), and LN-Frozen Counterfactual.
    """

    conditions = [
        "Clean\nBaseline",
        r"Cond. $\alpha$" + "\n(L9H9 ablated)",
        r"Cond. $\beta$" + "\n(L10H0 ablated)",
        r"Cond. $\gamma$" + "\n(Co-ablation)",
        "LN-Frozen\nCounterfactual",
    ]
    values = [0.3779, 0.5672, 0.3779, 0.5672, 0.5212]
    colors = [COL_CLEAN, COL_ALPHA, COL_BETA, COL_GAMMA, COL_LN_FROZEN]

    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    x = np.arange(len(conditions))
    bars = ax.bar(x, values, width=0.58, color=colors, edgecolor="white",
                  linewidth=0.8, zorder=3)

    # Add value labels on top of each bar
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color="#333333")

    # Horizontal reference line at clean baseline
    ax.axhline(y=0.3779, color=COL_CLEAN, linewidth=1.0, linestyle="--",
               alpha=0.6, zorder=2)
    ax.text(4.45, 0.3779 + 0.008, "Clean baseline", fontsize=8,
            color=COL_CLEAN, alpha=0.8, ha="right")

    # Delta annotations
    # α delta
    ax.annotate("", xy=(1, 0.5672), xytext=(1, 0.3779),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2))
    ax.text(1.32, 0.47, r"$\Delta = +0.1893$", fontsize=8.5, color="#444444",
            ha="left", va="center")

    # β delta = 0
    ax.annotate(r"$\Delta = 0.0000$", xy=(2, 0.3779), xytext=(2, 0.43),
                fontsize=8.5, color=COL_BETA, ha="center",
                arrowprops=dict(arrowstyle="->", color=COL_BETA, lw=1.0))

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=9.5)
    ax.set_ylabel("L10H10 Attention Probability to IO", fontsize=11.5)
    ax.set_ylim(0, 0.68)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    ax.set_title(
        "Sequential Isolation: L10H10 Self-Repair Trigger Identification",
        fontsize=13, fontweight="bold", pad=12
    )

    fig.tight_layout()
    fig.savefig("isolation_bar_chart.png")
    plt.close(fig)
    print("✓ Saved: isolation_bar_chart.png")


# ======================================================================
# FIGURE 2: QK Cosine Similarity Shift Lollipop Chart
# ======================================================================
def generate_cossim_shift_chart():
    """
    Lollipop chart showing the change in QK cosine similarity (Ablated - Clean)
    for every head in Layer 10. Positive shifts indicate active vector
    realignment toward the IO token.
    """

    # Load the CSV data
    df = pd.read_csv("fali_rigorous_validation.csv")
    df["Delta_CosSim"] = df["Abl_CosSim"] - df["Clean_CosSim"]
    df = df.sort_values("Delta_CosSim", ascending=True).reset_index(drop=True)

    heads = df["Head"].values
    deltas = df["Delta_CosSim"].values

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    y_pos = np.arange(len(heads))

    # Color each lollipop by sign
    colors = []
    for i, d in enumerate(deltas):
        if heads[i] == "L10H10":
            colors.append(COL_HIGHLIGHT)
        elif d >= 0:
            colors.append(COL_POS)
        else:
            colors.append(COL_NEG)

    # Stems
    for i, (yp, d, c) in enumerate(zip(y_pos, deltas, colors)):
        lw = 2.5 if heads[i] == "L10H10" else 1.8
        ax.plot([0, d], [yp, yp], color=c, linewidth=lw, zorder=2,
                solid_capstyle="round")

    # Dots
    for i, (yp, d, c) in enumerate(zip(y_pos, deltas, colors)):
        ms = 10 if heads[i] == "L10H10" else 7
        marker = "D" if heads[i] == "L10H10" else "o"
        ax.plot(d, yp, marker=marker, color=c, markersize=ms, zorder=3,
                markeredgecolor="white", markeredgewidth=0.8)

    # Value labels
    for i, (yp, d, h) in enumerate(zip(y_pos, deltas, heads)):
        offset = 0.004 if d >= 0 else -0.004
        ha = "left" if d >= 0 else "right"
        weight = "bold" if h == "L10H10" else "normal"
        ax.text(d + offset, yp, f"{d:+.4f}", va="center", ha=ha,
                fontsize=8.5, fontweight=weight, color="#333333")

    # Zero line
    ax.axvline(x=0, color="#888888", linewidth=0.8, linestyle="-", zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(heads, fontsize=10)
    ax.set_xlabel(
        r"$\Delta$ Cosine Similarity (Ablated $-$ Clean)",
        fontsize=11.5
    )
    ax.set_title(
        "QK Cosine Similarity Shifts: Geometric Evidence\n"
        "of Active Vector Realignment (Layer 10)",
        fontsize=13, fontweight="bold", pad=12
    )

    # Highlight L10H10 row with a subtle background band
    l10h10_idx = np.where(heads == "L10H10")[0]
    if len(l10h10_idx) > 0:
        idx = l10h10_idx[0]
        ax.axhspan(idx - 0.4, idx + 0.4, color=COL_HIGHLIGHT, alpha=0.08,
                   zorder=0)

    # Add a small legend-like annotation
    ax.text(0.97, 0.04,
            r"$\bullet$ Green = active realignment toward IO" "\n"
            r"$\bullet$ Red = degradation away from IO" "\n"
            u"\u25C6 L10H10 (primary backup head)",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    fig.tight_layout()
    fig.savefig("cossim_shift_lollipop.png")
    plt.close(fig)
    print("✓ Saved: cossim_shift_lollipop.png")


# ======================================================================
if __name__ == "__main__":
    print("Generating publication-quality figures...")
    generate_isolation_bar_chart()
    generate_cossim_shift_chart()
    print("\nAll figures generated successfully.")
