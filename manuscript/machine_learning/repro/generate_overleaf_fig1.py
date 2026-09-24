#!/usr/bin/env python3
"""Regenerate Figure 1 for the MLJ/Overleaf package.

This is a visual-only refinement: the scientific semantics are unchanged.
It fixes label/box overlap observed after rendering the compiled manuscript.
Outputs publication PDF/EPS plus editable SVG and high-resolution PNG.
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

MM = 1 / 25.4
W = 174 * MM


def build(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
    })
    factor_colors = {"T": "C0", "R": "C1", "O": "0.38", "V": "C2", "C": "C4"}

    fig = plt.figure(figsize=(W, 3.5))
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1.02, 0.98],
        width_ratios=[1.18, 0.82],
        hspace=0.28,
        wspace=0.22,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    for ax in (ax_a, ax_b, ax_c):
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    def panel(ax, label):
        ax.text(0.0, 1.02, label, fontsize=10, fontweight="bold", ha="left", va="bottom")

    def rbox(ax, x, y, w, h, text, ec="0.35", lw=1.0, fontsize=8.2, color="black"):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            ec=ec, fc="white", lw=lw,
            transform=ax.transAxes,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, color=color,
            transform=ax.transAxes,
        )

    def arrow(ax, x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=10,
            lw=0.9, color="0.25", transform=ax.transAxes,
        ))

    panel(ax_a, "(a)")
    panel(ax_b, "(b)")
    panel(ax_c, "(c)")

    for y, state, perturbed, effect in [
        (0.59, "Intact state", False, r"$E_A(h)$"),
        (0.16, "Perturbed state", True, r"$E_B(h)$"),
    ]:
        ax_a.text(0.06, y + 0.205, state, ha="left", va="bottom", fontsize=8.8, fontweight="bold")
        rbox(ax_a, 0.06, y, 0.28, 0.16, "Primary circuit", ec="0.45", fontsize=8.0)
        if perturbed:
            ax_a.plot([0.095, 0.305], [y + 0.025, y + 0.135], lw=1.7, color="C1", transform=ax_a.transAxes)
            ax_a.plot([0.305, 0.095], [y + 0.025, y + 0.135], lw=1.7, color="C1", transform=ax_a.transAxes)
        arrow(ax_a, 0.35, y + 0.08, 0.46, y + 0.08)
        rbox(ax_a, 0.47, y + 0.015, 0.17, 0.13, r"head $h$", ec="C0", lw=1.2, fontsize=8.5, color="C0")
        arrow(ax_a, 0.65, y + 0.08, 0.76, y + 0.08)
        ax_a.text(0.78, y + 0.08, effect, ha="left", va="center", fontsize=9.4)
    ax_a.text(0.06, 0.04, "primary perturbation", ha="left", fontsize=7.4, color="C1")

    ax_b.text(0.50, 0.72, r"$\Delta_h=E_B(h)-E_A(h)$", ha="center", va="center", fontsize=14.5)
    ax_b.text(0.50, 0.49, "Observed: usefulness changes", ha="center", va="center", fontsize=8.8)
    arrow(ax_b, 0.50, 0.43, 0.50, 0.34)
    rbox(ax_b, 0.18, 0.16, 0.64, 0.17, "Which changes receive credit?", ec="0.35", fontsize=8.4)
    ax_b.text(0.50, 0.08, "FALI", ha="center", va="center", fontsize=8.6, fontweight="bold")

    labels = [
        ("T", "target scores"),
        ("R", "rival scores"),
        ("O", "other valid scores"),
        ("V", "value content"),
        ("C", "surrounding context"),
    ]
    xs = np.linspace(0.095, 0.905, 5)
    for (lab, desc), x in zip(labels, xs):
        rbox(
            ax_c, x - 0.052, 0.53, 0.104, 0.20, lab,
            ec=factor_colors[lab], lw=1.2, fontsize=10.4, color=factor_colors[lab],
        )
        ax_c.text(x, 0.44, desc, ha="center", va="top", fontsize=7.7)
    ax_c.text(
        0.50, 0.15,
        r"$\phi_T+\phi_R+\phi_O+\phi_V+\phi_C=\Delta_h$",
        ha="center", va="center", fontsize=10.8,
    )

    for ext in ("pdf", "eps", "svg", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(out / f"Fig1.{ext}", **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    build(args.out_dir)
