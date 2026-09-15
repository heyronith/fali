"""
generate_new_figures.py

Generates compact publication figures for the FALI / TMLR submission:
  1. Sequential Isolation Bar Chart
  2. QK Cosine Similarity Shift Lollipop Chart
  3. Causal Shift Trajectory Scatter
  4. Circuit Disentanglement Scatter

Data sourced from fali_rigorous_validation.py / scale_evaluation.py outputs.
In-plot titles are omitted (LaTeX captions name the figures).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

COL_CLEAN = "#4C72B0"
COL_ALPHA = "#DD8452"
COL_BETA = "#55A868"
COL_GAMMA = "#C44E52"
COL_LN_FROZEN = "#8172B3"
COL_POS = "#2ca02c"
COL_NEG = "#d62728"
COL_HIGHLIGHT = "#ff7f0e"


def _ensure_figures_dir():
    os.makedirs("figures", exist_ok=True)


def generate_isolation_bar_chart():
    conditions = [
        "Clean\nBaseline",
        r"Cond. $\alpha$" + "\n(L9H9)",
        r"Cond. $\beta$" + "\n(L10H0)",
        r"Cond. $\gamma$" + "\n(Co-abl.)",
        "LN-Frozen\nCounterfact.",
    ]
    values = [0.3779, 0.5672, 0.3779, 0.5672, 0.5212]
    colors = [COL_CLEAN, COL_ALPHA, COL_BETA, COL_GAMMA, COL_LN_FROZEN]

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    x = np.arange(len(conditions))
    bars = ax.bar(x, values, width=0.58, color=colors, edgecolor="white",
                  linewidth=0.7, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.010,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color="#333333")

    ax.axhline(y=0.3779, color=COL_CLEAN, linewidth=0.9, linestyle="--",
               alpha=0.6, zorder=2)
    ax.annotate("", xy=(1, 0.5672), xytext=(1, 0.3779),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.0))
    ax.text(1.28, 0.47, r"$\Delta=+0.1893$", fontsize=10, color="#444444",
            ha="left", va="center")
    ax.annotate(r"$\Delta=0$", xy=(2, 0.3779), xytext=(2, 0.43),
                fontsize=10, color=COL_BETA, ha="center",
                arrowprops=dict(arrowstyle="->", color=COL_BETA, lw=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=12)
    ax.set_ylabel("L10H10 Attn Prob. to IO", fontsize=12)
    ax.set_ylim(0, 0.68)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    fig.tight_layout()
    _ensure_figures_dir()
    fig.savefig("figures/isolation_bar_chart.png")
    plt.close(fig)
    print("✓ Saved: figures/isolation_bar_chart.png")


def generate_cossim_shift_chart():
    df = pd.read_csv("results/summary/fali_rigorous_validation.csv")
    df["Delta_CosSim"] = df["Abl_CosSim"] - df["Clean_CosSim"]
    df = df.sort_values("Delta_CosSim", ascending=True).reset_index(drop=True)

    heads = df["Head"].values
    deltas = df["Delta_CosSim"].values

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    y_pos = np.arange(len(heads))

    colors = []
    for i, d in enumerate(deltas):
        if heads[i] == "L10H10":
            colors.append(COL_HIGHLIGHT)
        elif d >= 0:
            colors.append(COL_POS)
        else:
            colors.append(COL_NEG)

    for i, (yp, d, c) in enumerate(zip(y_pos, deltas, colors)):
        lw = 2.2 if heads[i] == "L10H10" else 1.5
        ax.plot([0, d], [yp, yp], color=c, linewidth=lw, zorder=2,
                solid_capstyle="round")

    for i, (yp, d, c) in enumerate(zip(y_pos, deltas, colors)):
        ms = 8 if heads[i] == "L10H10" else 5.5
        marker = "D" if heads[i] == "L10H10" else "o"
        ax.plot(d, yp, marker=marker, color=c, markersize=ms, zorder=3,
                markeredgecolor="white", markeredgewidth=0.6)

    for i, (yp, d, h) in enumerate(zip(y_pos, deltas, heads)):
        weight = "bold" if h == "L10H10" else "normal"
        if d >= 0:
            ax.text(d + 0.003, yp, f"{d:+.4f}", va="center", ha="left",
                    fontsize=10, fontweight=weight, color="#333333")
        else:
            ax.text(d + 0.008, yp + 0.28, f"{d:+.4f}", va="bottom", ha="left",
                    fontsize=10, fontweight=weight, color="#333333",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor="none", alpha=0.85))

    ax.axvline(x=0, color="#888888", linewidth=0.7, linestyle="-", zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(heads, fontsize=11)
    ax.set_xlabel(r"$\Delta$ CosSim (Ablated $-$ Clean)", fontsize=12)

    l10h10_idx = np.where(heads == "L10H10")[0]
    if len(l10h10_idx) > 0:
        idx = l10h10_idx[0]
        ax.axhspan(idx - 0.4, idx + 0.4, color=COL_HIGHLIGHT, alpha=0.08, zorder=0)

    ax.text(0.98, 0.03,
            r"$\bullet$ Green = $\uparrow$ IO alignment" "\n"
            r"$\bullet$ Red = $\downarrow$ IO alignment" "\n"
            r"$\diamond$ L10H10",
            transform=ax.transAxes, fontsize=9.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    fig.tight_layout()
    _ensure_figures_dir()
    fig.savefig("figures/cossim_shift_lollipop.png")
    plt.close(fig)
    print("✓ Saved: figures/cossim_shift_lollipop.png")


def generate_causal_shift_plot():
    """Compact causal-shift scatter from saved scale-evaluation JSON."""
    with open("results/summary/scale_evaluation_results.json", "r") as f:
        head_results = json.load(f)

    filtered = [r for r in head_results if r["layer"] in [9, 10, 11]]
    x_clean = [r["clean_attn"] for r in filtered]
    y_ablated = [r["standard_zero_ablation"] for r in filtered]
    names = [r["name"] for r in filtered]

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.grid(True, linestyle="--", alpha=0.55, zorder=1)

    ax.scatter(
        x_clean, y_ablated,
        c="lightgray", alpha=0.85, edgecolors="gray", s=42, zorder=2,
        label="Background",
    )

    primary_heads = {"L9H9", "L10H0"}
    backup_heads = {"L10H10", "L10H2", "L10H6", "L10H3"}

    for i, name in enumerate(names):
        if name in primary_heads:
            ax.scatter(
                x_clean[i], y_ablated[i], c="red", marker="*", s=160,
                edgecolors="darkred", zorder=4,
                label="Primary" if name == "L9H9" else "",
            )
            ax.annotate(
                name, (x_clean[i], y_ablated[i]), xytext=(4, 4),
                textcoords="offset points", fontsize=10, fontweight="bold",
                color="darkred",
            )
        elif name in backup_heads:
            ax.scatter(
                x_clean[i], y_ablated[i], c="limegreen", marker="^", s=95,
                edgecolors="darkgreen", zorder=4,
                label="Backup" if name == "L10H10" else "",
            )
            ax.annotate(
                name, (x_clean[i], y_ablated[i]), xytext=(4, 4),
                textcoords="offset points", fontsize=10, fontweight="bold",
                color="darkgreen",
            )
            ax.arrow(
                x_clean[i], x_clean[i],
                0, y_ablated[i] - x_clean[i],
                color="green", alpha=0.5, width=0.001,
                head_width=0.008, head_length=0.008,
                length_includes_head=True, zorder=3,
            )

    min_causal = min(min(x_clean), min(y_ablated), 0.0) - 0.02
    max_causal = max(max(x_clean), max(y_ablated), 0.3) + 0.05
    ax.plot(
        [min_causal, max_causal], [min_causal, max_causal],
        color="black", linestyle="--", linewidth=1.2, zorder=2,
        label="y = x",
    )
    ax.set_xlim(min_causal, max_causal)
    ax.set_ylim(min_causal, max_causal)
    ax.set_xlabel("Clean Attn Prob. to IO", fontsize=12)
    ax.set_ylabel("Ablated Attn Prob. to IO", fontsize=12)
    ax.legend(loc="upper left", frameon=True, fontsize=10.5)

    fig.tight_layout()
    _ensure_figures_dir()
    fig.savefig("figures/causal_shift_plot.png")
    plt.close(fig)
    print("✓ Saved: figures/causal_shift_plot.png")


def generate_circuit_disentanglement_plot():
    with open("results/summary/scale_evaluation_results.json", "r") as f:
        head_results = json.load(f)

    filtered = [r for r in head_results if r["layer"] in [9, 10, 11]]
    x_spikes = [r["mean_hydra_spike"] for r in filtered]
    y_spikes = [r["mean_hybrid_spike"] for r in filtered]
    layers = [r["layer"] for r in filtered]
    names = [r["name"] for r in filtered]

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.grid(True, linestyle="--", alpha=0.55, zorder=1)

    scatter = ax.scatter(
        x_spikes, y_spikes, c=layers, cmap="tab20",
        alpha=0.9, edgecolors="black", linewidths=0.5, s=56, zorder=3,
    )

    min_val = min(min(x_spikes), min(y_spikes), -0.05) - 0.05
    max_val = max(max(x_spikes), max(y_spikes), 0.25) + 0.05
    lims = [min_val, max_val]
    ax.plot(
        lims, lims, color="#4A4A4A", linestyle="--", linewidth=1.2,
        label="y = x", zorder=2,
    )
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xlabel("Standard Attention Spike", fontsize=12)
    ax.set_ylabel("FALI Target-Logit Contribution", fontsize=12)

    cbar = fig.colorbar(scatter, ax=ax, ticks=[9, 10, 11], fraction=0.046, pad=0.04)
    cbar.set_label("Layer", fontsize=11)

    annotated_heads = {"L10H10", "L11H10", "L9H9", "L10H0"}
    for i, name in enumerate(names):
        if name in annotated_heads or abs(x_spikes[i]) > 0.04:
            x_offset, y_offset = 0.004, 0.004
            if name == "L11H10":
                y_offset = -0.010
            elif name == "L10H10":
                x_offset = -0.038
            ax.annotate(
                name, (x_spikes[i], y_spikes[i]),
                xytext=(x_spikes[i] + x_offset, y_spikes[i] + y_offset),
                fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="yellow", alpha=0.5,
                          edgecolor="orange", lw=0.4),
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.65),
                zorder=4,
            )

    ax.legend(loc="upper left", frameon=True, fontsize=10.5)
    fig.tight_layout()
    _ensure_figures_dir()
    fig.savefig("figures/circuit_disentanglement_plot.png")
    plt.close(fig)
    print("✓ Saved: figures/circuit_disentanglement_plot.png")


if __name__ == "__main__":
    print("Generating compact publication figures...")
    generate_isolation_bar_chart()
    generate_cossim_shift_chart()
    generate_causal_shift_plot()
    generate_circuit_disentanglement_plot()
    print("\nAll figures generated successfully.")
