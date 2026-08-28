"""
Aggregate Phase-1 full multi-seed results and write scientific summary artifacts.

NOTE (Phase-1C):
  The legacy `recruitment_label` field in this script treats positive conditional
  *gain* as recruitment. That is incorrect when conditional_effect < 0
  ("less suppressive" is not recruitment).

  Corrected taxonomy lives in:
    experiments/revision_phase1_corrected_classification.py
    results/revision_phase1/full/aggregate_corrected/

  Do not use legacy labels for scientific claims about functional recruitment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from revision_config import BACKUP_NAME_MOVERS, head_name_from_pair
from revision_stats import pearson_spearman

REPO = Path(__file__).resolve().parents[1]
SEEDS = [42, 1, 8, 22]
HISTORICAL = [
    "results/summary/attention_results.json",
    "results/summary/ablation_controls_summary.json",
    "results/summary/output_mediation_summary.json",
    "results/raw/fali_decomposition_raw.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_row(mean_gain: float, seed_gains: List[float], seed_ci_signs: List[str]) -> str:
    """
    LEGACY label used only for provenance of Phase-1B aggregates.

    WARNING: This labels positive conditional *gain* as recruitment even when the
    head remains net-suppressive (conditional_effect < 0). Prefer the Phase-1C
    corrected classifier in revision_phase1_corrected_classification.py.
    """
    n_pos = sum(1 for g in seed_gains if g > 0)
    n_neg = sum(1 for g in seed_gains if g < 0)
    n_ci_pos = sum(1 for s in seed_ci_signs if s == "pos")
    n_ci_neg = sum(1 for s in seed_ci_signs if s == "neg")

    if mean_gain > 0 and n_pos >= 3 and n_ci_pos >= 3:
        return "LEGACY_positive_gain_only__not_functional_recruitment"
    if mean_gain > 0 and n_pos >= 3 and n_ci_pos < 3:
        return "LEGACY_suggestive_positive_gain_only__not_functional_recruitment"
    if mean_gain < 0 and n_neg >= 3 and n_ci_neg >= 3:
        return "LEGACY_stable_negative_conditional_gain"
    return "LEGACY_no_stable_gain"


def ci_sign(low: float, high: float) -> str:
    if low > 0:
        return "pos"
    if high < 0:
        return "neg"
    return "includes_zero"


def write_head_report(agg_row: pd.Series, seed_summaries: Dict[int, pd.DataFrame], path: Path) -> None:
    h = agg_row["head_name"]
    lines = [f"# {h} Phase-1 full-run report", "", "## Per-seed", ""]
    lines.append(
        "| seed | clean_attn | corrected_attn | spike | intact | conditional | gain | CI low | CI high | tmplA | tmplB |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for seed in SEEDS:
        s = seed_summaries[seed]
        r = s[s["head_name"] == h].iloc[0]
        lines.append(
            f"| {seed} | {r['mean_clean_attn']:.4f} | {r['mean_corrected_primary_attn']:.4f} | "
            f"{r['mean_corrected_attention_spike']:.4f} | {r['mean_intact_effect']:.4f} | "
            f"{r['mean_conditional_effect']:.4f} | {r['mean_conditional_gain']:.4f} | "
            f"{r['conditional_gain_ci_low']:.4f} | {r['conditional_gain_ci_high']:.4f} | "
            f"{r['template_A_conditional_gain']:.4f} | {r['template_B_conditional_gain']:.4f} |"
        )
    lines += [
        "",
        "## Across-seed means",
        "",
        f"- mean attention spike: {agg_row['mean_attention_spike_across_seeds']:.6f}",
        f"- mean intact effect: {agg_row['mean_intact_effect_across_seeds']:.6f}",
        f"- mean conditional effect: {agg_row['mean_conditional_effect_across_seeds']:.6f}",
        f"- mean conditional gain: {agg_row['mean_conditional_gain_across_seeds']:.6f}",
        f"- seeds with positive gain: {int(agg_row['n_seeds_positive_gain'])}/4",
        f"- seeds with positive CI excluding zero: {int(agg_row['n_seeds_gain_ci_excludes_zero_positive'])}/4",
        f"- classification: `{agg_row['recruitment_label']}`",
        "",
        "## Descriptive answer",
        "",
    ]
    label = agg_row["recruitment_label"]
    if label == "strong_positive_conditional_recruitment":
        lines.append(
            f"> Yes: `{h}` shows **strong positive conditional functional recruitment** "
            "under the corrected primary knockout by the pre-registered Phase-1 criteria."
        )
    elif label == "suggestive_positive_recruitment":
        lines.append(
            f"> Suggestive only: `{h}` has positive mean gain and ≥3/4 positive seeds, "
            "but fewer than 3 seed-level CIs exclude zero."
        )
    elif label == "stable_negative_conditional_gain":
        lines.append(
            f"> No: `{h}` shows **stable negative** conditional gain under the corrected knockout."
        )
    else:
        lines.append(
            f"> No stable positive recruitment for `{h}` under the corrected primary knockout "
            "by the Phase-1 criteria."
        )
    lines.append("")
    lines.append("This report does not discuss FALI.")
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--full-root",
        default="results/revision_phase1/full",
    )
    args = p.parse_args()
    full_root = REPO / args.full_root
    agg_dir = full_root / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    seed_summaries = {}
    seed_primary = {}
    seed_boot = {}
    seed_raw_paths = {}
    for seed in SEEDS:
        d = full_root / f"seed_{seed}"
        seed_summaries[seed] = pd.read_csv(d / "candidate_summary.csv")
        seed_primary[seed] = pd.read_csv(d / "primary_ablation_summary.csv")
        seed_boot[seed] = pd.read_csv(d / "bootstrap_summary.csv")
        seed_raw_paths[seed] = d / "candidate_raw.csv"

        # Sanity checks
        cfg = json.loads((d / "run_config.json").read_text())
        assert cfg["num_samples"] == 200
        assert cfg["PRIMARY_NAME_MOVERS"] == [[9, 6], [9, 9], [10, 0]]
        assert cfg["HISTORICAL_PRIMARY_NAME_MOVERS"] == [[9, 9], [10, 0]]
        meta = pd.read_csv(d / "dataset_metadata.csv")
        assert len(meta) == 200
        assert (meta["template_id"] == "A").sum() == 100
        assert (meta["template_id"] == "B").sum() == 100
        raw = pd.read_csv(d / "candidate_raw.csv")
        assert set(raw["head_name"]) == set(seed_summaries[seed]["head_name"])
        assert raw["head_name"].nunique() == 12
        assert np.isfinite(raw.select_dtypes(include=[np.number]).to_numpy()).all()
        for c in ["clean_attn", "historical_primary_attn", "corrected_primary_attn"]:
            assert ((raw[c] >= -1e-6) & (raw[c] <= 1 + 1e-6)).all()
        assert np.allclose(
            raw["conditional_gain"],
            (raw["B"] - raw["C_h"]) - (raw["A"] - raw["D_h"]),
        )

    # Aggregate candidates
    heads = seed_summaries[SEEDS[0]]["head_name"].tolist()
    rows = []
    for h in heads:
        role = seed_summaries[SEEDS[0]].loc[
            seed_summaries[SEEDS[0]]["head_name"] == h, "head_role"
        ].iloc[0]
        spikes, intact, cond, gains, fracs = [], [], [], [], []
        seed_gain_map = {}
        ci_signs = []
        n_ci_pos = 0
        n_ci_neg = 0
        tmplA, tmplB = [], []
        for seed in SEEDS:
            r = seed_summaries[seed][seed_summaries[seed]["head_name"] == h].iloc[0]
            spikes.append(r["mean_corrected_attention_spike"])
            intact.append(r["mean_intact_effect"])
            cond.append(r["mean_conditional_effect"])
            gains.append(r["mean_conditional_gain"])
            fracs.append(r["fraction_positive_conditional_gain"])
            seed_gain_map[seed] = float(r["mean_conditional_gain"])
            tmplA.append(r["template_A_conditional_gain"])
            tmplB.append(r["template_B_conditional_gain"])
            sign = ci_sign(r["conditional_gain_ci_low"], r["conditional_gain_ci_high"])
            ci_signs.append(sign)
            if sign == "pos":
                n_ci_pos += 1
            if sign == "neg":
                n_ci_neg += 1
        label = classify_row(float(np.mean(gains)), gains, ci_signs)
        rows.append(
            {
                "head_name": h,
                "head_role": role,
                "mean_attention_spike_across_seeds": float(np.mean(spikes)),
                "std_attention_spike_across_seeds": float(np.std(spikes, ddof=1)),
                "mean_intact_effect_across_seeds": float(np.mean(intact)),
                "std_intact_effect_across_seeds": float(np.std(intact, ddof=1)),
                "mean_conditional_effect_across_seeds": float(np.mean(cond)),
                "std_conditional_effect_across_seeds": float(np.std(cond, ddof=1)),
                "mean_conditional_gain_across_seeds": float(np.mean(gains)),
                "std_conditional_gain_across_seeds": float(np.std(gains, ddof=1)),
                "mean_fraction_positive_across_seeds": float(np.mean(fracs)),
                "n_seeds_positive_gain": int(sum(1 for g in gains if g > 0)),
                "n_seeds_gain_ci_excludes_zero_positive": n_ci_pos,
                "n_seeds_gain_ci_excludes_zero_negative": n_ci_neg,
                "seed42_gain": seed_gain_map[42],
                "seed1_gain": seed_gain_map[1],
                "seed8_gain": seed_gain_map[8],
                "seed22_gain": seed_gain_map[22],
                "template_A_mean_gain": float(np.mean(tmplA)),
                "template_B_mean_gain": float(np.mean(tmplB)),
                "recruitment_label": label,
            }
        )
    agg = pd.DataFrame(rows)
    agg.to_csv(agg_dir / "aggregate_candidate_summary.csv", index=False)

    ranking = agg.sort_values(
        "mean_conditional_gain_across_seeds", ascending=False
    ).copy()
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking.to_csv(agg_dir / "conditional_gain_ranking.csv", index=False)

    # Primary aggregate
    conditions = seed_primary[SEEDS[0]]["condition"].tolist()
    prow = []
    for cond_name in conditions:
        vals = []
        deg = []
        per_seed = {}
        for seed in SEEDS:
            r = seed_primary[seed][seed_primary[seed]["condition"] == cond_name].iloc[0]
            vals.append(r["mean_io_s"])
            deg.append(r["mean_abs_degradation_from_clean"])
            per_seed[seed] = float(r["mean_io_s"])
        prow.append(
            {
                "condition": cond_name,
                "mean_io_s_across_seeds": float(np.mean(vals)),
                "std_io_s_across_seeds": float(np.std(vals, ddof=1)),
                "mean_degradation_from_clean_across_seeds": float(np.mean(deg)),
                "seed42_io_s": per_seed[42],
                "seed1_io_s": per_seed[1],
                "seed8_io_s": per_seed[8],
                "seed22_io_s": per_seed[22],
            }
        )
    primary_agg = pd.DataFrame(prow)

    # Drop comparisons A - B_old and A - B
    drop_rows = []
    for seed in SEEDS:
        p = seed_primary[seed].set_index("condition")
        A = float(p.loc["A_clean", "mean_io_s"])
        Bold = float(p.loc["B_old_historical", "mean_io_s"])
        B = float(p.loc["B_corrected", "mean_io_s"])
        hist = A - Bold
        corr = A - B
        drop_rows.append(
            {
                "seed": seed,
                "A": A,
                "B_old": Bold,
                "B": B,
                "historical_primary_drop": hist,
                "corrected_primary_drop": corr,
                "difference_in_drop": corr - hist,
            }
        )
    drop_df = pd.DataFrame(drop_rows)
    drop_df.to_csv(agg_dir / "primary_drop_by_seed.csv", index=False)
    primary_agg.to_csv(agg_dir / "aggregate_primary_summary.csv", index=False)

    # Correlations
    corr_rows = []
    for subset_name, mask in [
        ("all_12_candidates", np.ones(len(agg), dtype=bool)),
        ("backup_heads_only", agg["head_role"].eq("backup").to_numpy()),
    ]:
        sub = agg.loc[mask]
        c = pearson_spearman(
            sub["mean_attention_spike_across_seeds"],
            sub["mean_conditional_gain_across_seeds"],
        )
        corr_rows.append({"subset": subset_name, "n": int(mask.sum()), **c})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(agg_dir / "aggregate_correlation_summary.csv", index=False)

    # Scatter
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    colors = {
        "backup": "#1f77b4",
        "negative": "#d62728",
        "extra_control": "#7f7f7f",
    }
    for _, r in agg.iterrows():
        ax.scatter(
            r["mean_attention_spike_across_seeds"],
            r["mean_conditional_gain_across_seeds"],
            c=colors.get(r["head_role"], "black"),
            s=70,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.annotate(
            r["head_name"],
            (
                r["mean_attention_spike_across_seeds"],
                r["mean_conditional_gain_across_seeds"],
            ),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("Mean corrected attention spike (across seeds)")
    ax.set_ylabel("Mean conditional gain (across seeds)")
    ax.set_title("Attention spike vs conditional gain (descriptive)")
    fig.tight_layout()
    fig.savefig(agg_dir / "attention_spike_vs_conditional_gain.png", dpi=200)
    plt.close(fig)

    # Head reports
    for h in ["L10H10", "L10H2"]:
        row = agg[agg["head_name"] == h].iloc[0]
        write_head_report(row, seed_summaries, agg_dir / f"{h}_report.md")

    # Historical integrity
    hist = {rel: sha256(REPO / rel) for rel in HISTORICAL}
    (agg_dir / "historical_hashes_after_phase1b.json").write_text(json.dumps(hist, indent=2))

    print("Wrote aggregate artifacts to", agg_dir)
    print(agg[["head_name", "mean_conditional_gain_across_seeds", "recruitment_label"]].to_string(index=False))
    print(corr_df.to_string(index=False))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(REPO / "src"))
    main()
