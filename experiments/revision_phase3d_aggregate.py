"""
Phase-3D aggregation: factorial Shapley + interactions + Phase-2 comparison.

Reads results/revision_phase3c/full/seed_*/factorial_raw.csv
Writes results/revision_phase3c/full/aggregate/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from revision_config import BACKUP_NAME_MOVERS, head_name_from_pair  # noqa: E402
from revision_stats import pearson_spearman, stratified_bootstrap_ci  # noqa: E402

SEEDS = [42, 1, 8, 22]
N_BOOT = 10000
BACKUP_NAMES = [head_name_from_pair(h) for h in BACKUP_NAME_MOVERS]

BOOT_METRICS = [
    "conditional_gain",
    "phi_pattern",
    "phi_value",
    "phi_context",
    "main_P",
    "main_V",
    "main_C",
    "interaction_PV",
    "interaction_PC",
    "interaction_VC",
    "interaction_PVC",
    "pattern_effect_B_VB",
    "value_effect_B_PB",
    "context_effect_PB_VB",
]


def stable_str_hash_int(s: str, mod: int = 10_000_000) -> int:
    digest = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % mod


def boot_seed(base: int, seed: int, head: str, metric: str) -> int:
    return (
        base
        + 1009 * int(seed)
        + 17 * stable_str_hash_int(str(head), 10000)
        + 3 * stable_str_hash_int(str(metric), 10000)
    ) % (2**31 - 1)


def load_all(full_root: Path) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        p = full_root / f"seed_{seed}" / "factorial_raw.csv"
        df = pd.read_csv(p)
        assert len(df) == 200 * 12, f"seed {seed}: {len(df)}"
        assert df["head_name"].nunique() == 12
        n_a = int((df.drop_duplicates("example_idx")["template_id"] == "A").sum())
        n_b = int((df.drop_duplicates("example_idx")["template_id"] == "B").sum())
        assert n_a == 100 and n_b == 100
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    assert len(out) == 9600
    return out


def head_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "intact_effect",
        "conditional_effect",
        "conditional_gain",
        "phi_pattern",
        "phi_value",
        "phi_context",
        "main_P",
        "main_V",
        "main_C",
        "interaction_PV",
        "interaction_PC",
        "interaction_VC",
        "interaction_PVC",
        "pattern_effect_B_VB",
        "value_effect_B_PB",
        "context_effect_PB_VB",
        "pattern_effect_A_VA",
        "value_effect_A_PA",
        "context_effect_PA_VA",
        "existing_end_query_causal_total",
        "shapley_residual",
        "interaction_recon_residual",
        "err_H000_vs_intact",
        "err_H111_vs_conditional",
    ]
    g = (
        df.groupby(["seed", "head_name", "head_role"], sort=False)[metrics]
        .mean()
        .reset_index()
    )
    return g


def head_aggregate(hs: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in hs.columns if c not in ("seed", "head_name", "head_role")]
    rows = []
    for (head, role), sub in hs.groupby(["head_name", "head_role"], sort=False):
        row = {"head_name": head, "head_role": role, "n_seeds": len(sub)}
        for m in metrics:
            vals = sub[m].to_numpy(dtype=np.float64)
            row[f"mean_{m}"] = float(np.mean(vals))
            row[f"std_{m}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"n_seeds_pos_{m}"] = int(np.sum(vals > 0))
            row[f"n_seeds_neg_{m}"] = int(np.sum(vals < 0))
        rows.append(row)
    return pd.DataFrame(rows)


def make_bootstrap(df: pd.DataFrame, base: int = 42) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for head, sub in df[df["seed"] == seed].groupby("head_name"):
            strata = sub["template_id"].to_numpy()
            role = sub["head_role"].iloc[0]
            for metric in BOOT_METRICS:
                vals = sub[metric].to_numpy(dtype=np.float64)
                mean, lo, hi = stratified_bootstrap_ci(
                    vals,
                    strata,
                    n_bootstrap=N_BOOT,
                    seed=boot_seed(base, seed, head, metric),
                )
                rows.append(
                    {
                        "seed": seed,
                        "head_name": head,
                        "head_role": role,
                        "metric": metric,
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "ci_excludes_zero_pos": lo > 0,
                        "ci_excludes_zero_neg": hi < 0,
                        "n": len(vals),
                        "n_bootstrap": N_BOOT,
                    }
                )
    return pd.DataFrame(rows)


def template_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "conditional_gain",
        "phi_pattern",
        "phi_value",
        "phi_context",
        "pattern_effect_B_VB",
        "intact_effect",
        "conditional_effect",
    ]
    return (
        df.groupby(["seed", "template_id", "head_name", "head_role"], sort=False)[metrics]
        .mean()
        .reset_index()
    )


def flag_template_sign_reversals(ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["conditional_gain", "phi_pattern", "phi_value", "phi_context"]
    for seed in SEEDS:
        for head, sub in ts[ts["seed"] == seed].groupby("head_name"):
            a = sub[sub["template_id"] == "A"]
            b = sub[sub["template_id"] == "B"]
            if len(a) != 1 or len(b) != 1:
                continue
            for m in metrics:
                va, vb = float(a[m].iloc[0]), float(b[m].iloc[0])
                rev = (va > 0 and vb < 0) or (va < 0 and vb > 0)
                rows.append(
                    {
                        "seed": seed,
                        "head_name": head,
                        "metric": m,
                        "template_A": va,
                        "template_B": vb,
                        "sign_reversal": rev,
                    }
                )
    return pd.DataFrame(rows)


def phase2_comparison(hs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        sub = hs[hs["seed"] == seed]
        for _, r in sub.iterrows():
            end_q = float(r["existing_end_query_causal_total"])
            full_p = float(r["pattern_effect_B_VB"])
            rows.append(
                {
                    "seed": seed,
                    "head_name": r["head_name"],
                    "head_role": r["head_role"],
                    "pattern_effect_B_VB": full_p,
                    "end_query_causal_total": end_q,
                    "diff_full_minus_end": full_p - end_q,
                    "abs_diff": abs(full_p - end_q),
                    "phi_pattern": float(r["phi_pattern"]),
                    "diff_phi_minus_end": float(r["phi_pattern"]) - end_q,
                }
            )
    per = pd.DataFrame(rows)
    # correlations across heads per seed + seedmean
    corr_rows = []
    for seed in SEEDS:
        sub = per[per["seed"] == seed]
        corr = pearson_spearman(sub["pattern_effect_B_VB"], sub["end_query_causal_total"])
        corr_rows.append({"level": "across_head", "seed": seed, **corr})
    agg = (
        per.groupby("head_name", sort=False)[
            ["pattern_effect_B_VB", "end_query_causal_total"]
        ]
        .mean()
        .reset_index()
    )
    corr = pearson_spearman(agg["pattern_effect_B_VB"], agg["end_query_causal_total"])
    corr_rows.append({"level": "across_head_seedmean", "seed": "all", **corr})
    return per, pd.DataFrame(corr_rows)


def mechanism_function_corr(hs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, names in [("backups", BACKUP_NAMES), ("all12", None)]:
        for seed in list(SEEDS) + ["seedmean"]:
            if seed == "seedmean":
                sub = hs.groupby("head_name", sort=False).mean(numeric_only=True).reset_index()
                sub = sub.merge(
                    hs[["head_name", "head_role"]].drop_duplicates(), on="head_name"
                )
            else:
                sub = hs[hs["seed"] == seed].copy()
            if names is not None:
                sub = sub[sub["head_name"].isin(names)]
            for x in ["phi_pattern", "phi_context", "phi_value"]:
                corr = pearson_spearman(sub[x], sub["conditional_gain"])
                rows.append(
                    {
                        "scope": scope,
                        "seed": seed,
                        "x": x,
                        "y": "conditional_gain",
                        **corr,
                    }
                )
    return pd.DataFrame(rows)


def value_importance(agg: pd.DataFrame) -> Dict:
    ap = float(agg["mean_phi_pattern"].abs().mean())
    av = float(agg["mean_phi_value"].abs().mean())
    ac = float(agg["mean_phi_context"].abs().mean())
    return {
        "mean_abs_phi_pattern": ap,
        "mean_abs_phi_value": av,
        "mean_abs_phi_context": ac,
        "value_over_pattern": av / ap if ap > 1e-12 else float("nan"),
        "value_over_context": av / ac if ac > 1e-12 else float("nan"),
    }


def ci_sign_consistency(boot: pd.DataFrame, head: str, metric: str) -> Dict:
    sub = boot[(boot["head_name"] == head) & (boot["metric"] == metric)]
    n_pos = int(sub["ci_excludes_zero_pos"].sum())
    n_neg = int(sub["ci_excludes_zero_neg"].sum())
    means = sub["mean"].to_numpy()
    return {
        "mean_across_seeds": float(np.mean(means)),
        "n_ci_pos": n_pos,
        "n_ci_neg": n_neg,
        "n_mean_pos": int(np.sum(means > 0)),
        "n_mean_neg": int(np.sum(means < 0)),
    }


def write_head_report(
    path: Path,
    title: str,
    agg: pd.DataFrame,
    boot: pd.DataFrame,
    head: str,
    body: str,
) -> None:
    r = agg[agg["head_name"] == head].iloc[0]
    lines = [
        f"# {title}",
        "",
        f"Head: **{head}** (means across 4 seeds, N=200 each).",
        "",
        f"- intact_effect: {r['mean_intact_effect']:.4f}",
        f"- conditional_effect: {r['mean_conditional_effect']:.4f}",
        f"- conditional_gain: {r['mean_conditional_gain']:.4f}",
        f"- phi_pattern: {r['mean_phi_pattern']:.4f} (pos seeds {r['n_seeds_pos_phi_pattern']}/4)",
        f"- phi_value: {r['mean_phi_value']:.4f}",
        f"- phi_context: {r['mean_phi_context']:.4f} (pos seeds {r['n_seeds_pos_phi_context']}/4)",
        f"- pattern_effect_B_VB: {r['mean_pattern_effect_B_VB']:.4f}",
        f"- interaction_PC: {r['mean_interaction_PC']:.4f}",
        "",
        "## Bootstrap CI consistency",
    ]
    for m in ["conditional_gain", "phi_pattern", "phi_value", "phi_context"]:
        c = ci_sign_consistency(boot, head, m)
        lines.append(
            f"- {m}: mean={c['mean_across_seeds']:.4f}; "
            f"CI>0 in {c['n_ci_pos']}/4; CI<0 in {c['n_ci_neg']}/4"
        )
    lines += ["", body, ""]
    path.write_text("\n".join(lines))


def make_plots(agg: pd.DataFrame, hs: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    heads = agg["head_name"].tolist()
    x = np.arange(len(heads))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, agg["mean_phi_pattern"], width, label="phi_pattern")
    ax.bar(x, agg["mean_phi_value"], width, label="phi_value")
    ax.bar(x + width, agg["mean_phi_context"], width, label="phi_context")
    ax.set_xticks(x)
    ax.set_xticklabels(heads, rotation=45, ha="right")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Shapley contribution to conditional_gain")
    ax.set_title("conditional_gain decomposition (seed-mean)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "gain_decomposition_all_heads.png", dpi=160)
    plt.close(fig)

    # L10H2 vs L10H10
    panel = agg[agg["head_name"].isin(["L10H2", "L10H10"])]
    fig, ax = plt.subplots(figsize=(6, 4))
    labs = panel["head_name"].tolist()
    xp = np.arange(len(labs))
    ax.bar(xp - width, panel["mean_phi_pattern"], width, label="phi_pattern")
    ax.bar(xp, panel["mean_phi_value"], width, label="phi_value")
    ax.bar(xp + width, panel["mean_phi_context"], width, label="phi_context")
    ax.set_xticks(xp)
    ax.set_xticklabels(labs)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("L10H2 vs L10H10 decomposition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "L10H2_vs_L10H10_decomposition.png", dpi=160)
    plt.close(fig)

    seedmean = hs.groupby("head_name", sort=False).mean(numeric_only=True).reset_index()

    def scatter(xcol, ycol, path, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(seedmean[xcol], seedmean[ycol], s=60, c="#1f4e79")
        for _, r in seedmean.iterrows():
            ax.annotate(
                r["head_name"],
                (r[xcol], r[ycol]),
                fontsize=8,
                xytext=(4, 4),
                textcoords="offset points",
            )
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    scatter(
        "phi_pattern",
        "conditional_gain",
        out / "phi_pattern_vs_conditional_gain.png",
        "phi_pattern",
        "conditional_gain",
    )
    scatter(
        "phi_context",
        "conditional_gain",
        out / "phi_context_vs_conditional_gain.png",
        "phi_context",
        "conditional_gain",
    )
    scatter(
        "existing_end_query_causal_total",
        "pattern_effect_B_VB",
        out / "endquery_vs_full_pattern_B_VB.png",
        "Phase-2 end-query causal_total",
        "full-seq pattern_effect_B_VB",
    )


def classify_verdict(
    agg: pd.DataFrame,
    boot: pd.DataFrame,
    rev: pd.DataFrame,
) -> Tuple[str, List[str]]:
    notes = []

    def ok_pos(head, metric, min_ci=3):
        c = ci_sign_consistency(boot, head, metric)
        return c["n_ci_pos"] >= min_ci and c["mean_across_seeds"] > 0

    def ok_neg(head, metric, min_ci=3):
        c = ci_sign_consistency(boot, head, metric)
        return c["n_ci_neg"] >= min_ci and c["mean_across_seeds"] < 0

    def near_zero(head, metric, atol=1e-4):
        c = ci_sign_consistency(boot, head, metric)
        return abs(c["mean_across_seeds"]) <= atol

    h1 = near_zero("L9H7", "phi_pattern") and near_zero("L9H7", "phi_value") and ok_pos(
        "L9H7", "phi_context"
    )
    notes.append(f"H1 L9H7 context recruitment: {h1}")

    h2 = ok_pos("L10H2", "conditional_gain") and ok_pos("L10H2", "phi_pattern")
    notes.append(f"H2 L10H2 pattern-mediated +gain: {h2}")

    h3 = (
        ok_pos("L10H10", "phi_pattern")
        and ok_neg("L10H10", "phi_context")
        and ok_neg("L10H10", "conditional_gain")
    )
    notes.append(f"H3 L10H10 cancellation: {h3}")

    h4_gain = ok_pos("L10H7", "conditional_gain") or (
        ci_sign_consistency(boot, "L10H7", "conditional_gain")["mean_across_seeds"] > 0
        and ci_sign_consistency(boot, "L10H7", "phi_pattern")["mean_across_seeds"] > 0
    )
    h4_sup = (
        agg.loc[agg["head_name"] == "L10H7", "mean_conditional_effect"].iloc[0] < 0
    )
    h4 = h4_sup and (
        ok_pos("L10H7", "phi_pattern")
        or ci_sign_consistency(boot, "L10H7", "phi_pattern")["n_mean_pos"] >= 3
    )
    notes.append(f"H4 L10H7 adaptive-but-suppressive: {h4} (cond_effect<0={h4_sup})")

    # Template agreement: major heads should not reverse on gain/phi_pattern/phi_context
    major = ["L9H7", "L10H2", "L10H10", "L10H7"]
    major_rev = rev[
        rev["head_name"].isin(major)
        & rev["metric"].isin(["conditional_gain", "phi_pattern", "phi_context"])
        & rev["sign_reversal"]
    ]
    template_ok = len(major_rev) == 0
    notes.append(f"Template A/B major-head sign reversals: {len(major_rev)}")

    contrasts = [h1, h2, h3, h4]
    n_ok = sum(1 for x in contrasts if x)
    notes.append(f"contrasts passed: {n_ok}/4; template_ok={template_ok}")

    if n_ok == 4 and template_ok:
        return "STRONG REPLACEMENT", notes
    if n_ok >= 2:
        return "PARTIAL", notes
    return "FAILED PIVOT", notes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full-root", default="results/revision_phase3c/full")
    p.add_argument("--out-dir", default="results/revision_phase3c/full/aggregate")
    p.add_argument("--boot-seed", type=int, default=42)
    args = p.parse_args()

    full_root = REPO / args.full_root
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)

    print("Loading factorial raw...")
    df = load_all(full_root)
    df.to_csv(out / "factorial_merged_raw.csv", index=False)

    print("Summaries...")
    hs = head_seed_summary(df)
    hs.to_csv(out / "factorial_head_seed_summary.csv", index=False)
    agg = head_aggregate(hs)
    agg.to_csv(out / "factorial_head_aggregate_summary.csv", index=False)

    inter_cols = [
        "head_name",
        "head_role",
        "mean_main_P",
        "mean_main_V",
        "mean_main_C",
        "mean_interaction_PV",
        "mean_interaction_PC",
        "mean_interaction_VC",
        "mean_interaction_PVC",
    ]
    agg[inter_cols].to_csv(out / "factorial_interactions.csv", index=False)

    marg_cols = [
        "head_name",
        "head_role",
        "mean_pattern_effect_B_VB",
        "mean_value_effect_B_PB",
        "mean_context_effect_PB_VB",
        "mean_pattern_effect_A_VA",
        "mean_value_effect_A_PA",
        "mean_context_effect_PA_VA",
    ]
    # head_aggregate prefixes mean_
    marg_df = agg[
        [
            "head_name",
            "head_role",
            "mean_pattern_effect_B_VB",
            "mean_value_effect_B_PB",
            "mean_context_effect_PB_VB",
            "mean_pattern_effect_A_VA",
            "mean_value_effect_A_PA",
            "mean_context_effect_PA_VA",
        ]
    ]
    marg_df.to_csv(out / "endpoint_marginal_effects.csv", index=False)

    print("Bootstrap...")
    boot = make_bootstrap(df, base=args.boot_seed)
    boot.to_csv(out / "factorial_bootstrap_summary.csv", index=False)

    print("Templates...")
    ts = template_summary(df)
    ts.to_csv(out / "template_summary.csv", index=False)
    rev = flag_template_sign_reversals(ts)
    rev.to_csv(out / "template_sign_reversals.csv", index=False)

    print("Phase-2 comparison...")
    per, corr = phase2_comparison(hs)
    per.to_csv(out / "phase2_endquery_vs_fullpattern.csv", index=False)
    corr.to_csv(out / "phase2_endquery_vs_fullpattern_corr.csv", index=False)

    mfc = mechanism_function_corr(hs)
    mfc.to_csv(out / "mechanism_vs_function_correlations.csv", index=False)

    val_imp = value_importance(agg)

    write_head_report(
        out / "L9H7_context_report.md",
        "H1 — L9H7 context recruitment",
        agg,
        boot,
        "L9H7",
        "Prediction: phi_pattern≈0, phi_value≈0, phi_context>0. Same-layer P/V should be zero.",
    )
    write_head_report(
        out / "L10H2_pattern_report.md",
        "H2 — L10H2 pattern-mediated recruitment",
        agg,
        boot,
        "L10H2",
        "Prediction: conditional_gain>0 and phi_pattern>0 explaining substantial absolute gain.",
    )
    write_head_report(
        out / "L10H10_cancellation_report.md",
        "H3 — L10H10 cancellation",
        agg,
        boot,
        "L10H10",
        "Prediction: phi_pattern>0, phi_context<0, conditional_gain<0.",
    )
    write_head_report(
        out / "L10H7_suppressor_report.md",
        "H4 — L10H7 adaptive-but-suppressive",
        agg,
        boot,
        "L10H7",
        "Prediction: large positive pattern-mediated change with conditional_effect<0.",
    )
    write_head_report(
        out / "L10H1_signflip_report.md",
        "L10H1 suggestive sign-flip",
        agg,
        boot,
        "L10H1",
        "Assess whether intact→conditional sign change associates with pattern/value/context/interactions. "
        "Do not call the flip strong unless CIs support it.",
    )

    print("Plots...")
    make_plots(agg, hs, plots)

    verdict, notes = classify_verdict(agg, boot, rev)

    # Heterogeneity note
    backups = agg[agg["head_name"].isin(BACKUP_NAMES)]
    mech_notes = []
    for _, r in backups.iterrows():
        parts = []
        if abs(r["mean_phi_pattern"]) >= abs(r["mean_phi_context"]) and abs(
            r["mean_phi_pattern"]
        ) > 1e-3:
            parts.append("pattern")
        if abs(r["mean_phi_context"]) > abs(r["mean_phi_pattern"]) and abs(
            r["mean_phi_context"]
        ) > 1e-3:
            parts.append("context")
        if abs(r["mean_phi_pattern"]) > 1e-3 and abs(r["mean_phi_context"]) > 1e-3:
            parts = ["mixed"]
        if abs(r["mean_phi_pattern"]) <= 1e-3 and abs(r["mean_phi_context"]) > 1e-3:
            parts = ["context"]
        if abs(r["mean_phi_context"]) <= 1e-3 and abs(r["mean_phi_pattern"]) > 1e-3:
            parts = ["pattern"]
        mech_notes.append(f"{r['head_name']}: {'/'.join(parts) if parts else 'near-zero'}")

    sci = [
        "# Phase-3D scientific summary",
        "",
        f"**Predefined verdict: {verdict}**",
        "",
        "## Notes",
    ]
    for n in notes:
        sci.append(f"- {n}")
    sci += ["", "## Value factor magnitude", json.dumps(val_imp, indent=2)]
    sci += ["", "## Backup mechanism sketch (descriptive, not forced labels)"]
    for m in mech_notes:
        sci.append(f"- {m}")
    sci += [
        "",
        "## Phase-2 end-query vs full-sequence pattern_effect_B_VB",
        corr.to_string(index=False),
        "",
        "## Aggregate decomposition (selected)",
        agg[
            [
                "head_name",
                "mean_conditional_gain",
                "mean_phi_pattern",
                "mean_phi_value",
                "mean_phi_context",
                "mean_pattern_effect_B_VB",
            ]
        ].to_string(index=False),
        "",
        "## Rules",
        "- STRONG REPLACEMENT: H1–H4 reproduce with template-stable signs",
        "- PARTIAL: some contrasts only",
        "- FAILED PIVOT: smoke effects disappear / unstable",
    ]
    (out / "phase3d_scientific_summary.md").write_text("\n".join(sci) + "\n")

    meta = {
        "n_rows": len(df),
        "verdict": verdict,
        "notes": notes,
        "value_importance": val_imp,
        "n_bootstrap": N_BOOT,
        "template_major_reversals": int(
            rev[
                rev["head_name"].isin(["L9H7", "L10H2", "L10H10", "L10H7"])
                & rev["sign_reversal"]
            ].shape[0]
        ),
    }
    (out / "aggregate_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"VERDICT: {verdict}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
