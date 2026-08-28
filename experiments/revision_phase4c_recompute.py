"""
Phase-4C: fix hierarchical bootstrap CIs from existing Phase-4B raw factorial only.

No model execution. Does not alter raw factorial CSVs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

import revision_phase4_aggregate as agg  # noqa: E402

FULL = REPO / "results/revision_phase4/full"
AGG = FULL / "aggregate"
PRE = FULL / "aggregate_pre_bootstrap_fix"
ROLE_NEAR_ZERO = 0.05  # |effect| below this → near_zero/mixed (documented)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_role(cond: float, gain: float, tol: float = ROLE_NEAR_ZERO) -> str:
    if abs(cond) < tol or abs(gain) < tol:
        return "near_zero_or_mixed"
    if cond > 0 and gain > 0:
        return "supportive_recruited"
    if cond > 0 and gain < 0:
        return "supportive_downregulated"
    if cond < 0 and gain > 0:
        return "suppressive_less_suppressive"
    if cond < 0 and gain < 0:
        return "suppressive_more_suppressive"
    return "near_zero_or_mixed"


def backup_pre_fix() -> dict:
    PRE.mkdir(parents=True, exist_ok=True)
    keys = [
        "hierarchical_template_cluster_summary.csv",
        "hierarchical_template_cluster_nonoverlap29.csv",
        "hypothesis_evaluation.csv",
        "scientific_verdict.md",
        "L9H7_broad_template_report.md",
        "L10H2_broad_template_report.md",
        "L10H10_broad_template_report.md",
        "L10H7_broad_template_report.md",
        "L10H1_broad_template_report.md",
    ]
    hashes = {}
    for name in keys:
        src = AGG / name
        if src.exists():
            dst = PRE / name
            shutil.copy2(src, dst)
            hashes[name] = file_sha256(src)
    (PRE / "pre_fix_hashes.json").write_text(json.dumps(hashes, indent=2))
    return hashes


def recompute_hierarchical(df: pd.DataFrame, df29: pd.DataFrame):
    print("Recomputing hierarchical template-cluster (fixed)...", flush=True)
    hier = agg.hierarchical_cluster_summary(df)
    hier.to_csv(AGG / "hierarchical_template_cluster_summary.csv", index=False)

    print("Recomputing nonoverlap29 hierarchical...", flush=True)
    df_paired29 = df[df["low_level_template_id"] != 0].copy()
    hier_29_rows = []
    for hname, sub in df29.groupby("head_name", sort=False):
        role = sub["head_role"].iloc[0]
        for metric in agg.PRIMARY_BOOT_METRICS:
            seed_means = (
                sub.groupby("seed")[metric].mean().reindex(agg.SEEDS).to_numpy(dtype=np.float64)
            )
            point = float(np.mean(seed_means))
            sub28 = df_paired29[df_paired29["head_name"] == hname]
            tid = sub28["low_level_template_id"].to_numpy().astype(np.int64) - 1
            from revision_stats import hierarchical_template_cluster_bootstrap_ci

            _, lo, hi = hierarchical_template_cluster_bootstrap_ci(
                sub28[metric].to_numpy(dtype=np.float64),
                sub28["seed"].to_numpy(),
                tid,
                sub28["pattern_type"].to_numpy(),
                n_bootstrap=agg.N_BOOT,
                seed=agg.boot_seed("hier29", hname, metric),
                n_templates=14,
            )
            hier_29_rows.append(
                {
                    "head_name": hname,
                    "head_role": role,
                    "metric": metric,
                    "aggregate_mean": point,
                    "ci_low": lo,
                    "ci_high": hi,
                    "seed_mean_42": float(seed_means[0]),
                    "seed_mean_1": float(seed_means[1]),
                    "seed_mean_8": float(seed_means[2]),
                    "seed_mean_22": float(seed_means[3]),
                    "n_seeds_pos": int(np.sum(seed_means > 0)),
                    "n_seeds_neg": int(np.sum(seed_means < 0)),
                    "procedure": "hierarchical_14paired_ci_29point",
                    "n_bootstrap": agg.N_BOOT,
                }
            )
    hier_29 = pd.DataFrame(hier_29_rows)
    hier_29.to_csv(AGG / "hierarchical_template_cluster_nonoverlap29.csv", index=False)
    return hier, hier_29


def compare_bootstrap(old_hyp: pd.DataFrame, new_hyp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, old in old_hyp.iterrows():
        new = new_hyp[new_hyp["hypothesis_id"] == old["hypothesis_id"]].iloc[0]
        old_w = float(old["hierarchical_cluster_ci_high"] - old["hierarchical_cluster_ci_low"])
        new_w = float(new["hierarchical_cluster_ci_high"] - new["hierarchical_cluster_ci_low"])
        rows.append(
            {
                "hypothesis_id": old["hypothesis_id"],
                "head_name": old["head_name"],
                "metric": old["metric"],
                "old_mean": float(old["aggregate_mean"]),
                "old_ci_low": float(old["hierarchical_cluster_ci_low"]),
                "old_ci_high": float(old["hierarchical_cluster_ci_high"]),
                "new_mean": float(new["aggregate_mean"]),
                "new_ci_low": float(new["hierarchical_cluster_ci_low"]),
                "new_ci_high": float(new["hierarchical_cluster_ci_high"]),
                "CI_width_change": new_w - old_w,
                "mean_abs_diff": abs(float(new["aggregate_mean"]) - float(old["aggregate_mean"])),
                "old_status": old["all30_status"],
                "new_status": new["all30_status"],
                "status_changed": str(old["all30_status"]) != str(new["all30_status"]),
                "old_nonoverlap29_status": old["nonoverlap29_status"],
                "new_nonoverlap29_status": new["nonoverlap29_status"],
                "nonoverlap_status_changed": str(old["nonoverlap29_status"])
                != str(new["nonoverlap29_status"]),
            }
        )
    return pd.DataFrame(rows)


def functional_role_by_stratum(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(
            [
                "head_name",
                "head_role",
                "low_level_template_id",
                "pattern_type",
                "template_stratum",
            ],
            sort=True,
        )[["intact_effect", "conditional_effect", "conditional_gain"]]
        .mean()
        .reset_index()
    )
    g["role_tol"] = ROLE_NEAR_ZERO
    g["functional_role"] = [
        classify_role(float(r.conditional_effect), float(r.conditional_gain))
        for r in g.itertuples()
    ]
    # Provenance taxonomy (historical Phase-1 label) — separate from measured role
    g["historical_taxonomy_role"] = g["head_role"]
    return g


def baba_abba_functional_roles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    heads = sorted(df["head_name"].unique())
    for hname in heads:
        hsub = df[df["head_name"] == hname]
        hist = hsub["head_role"].iloc[0]
        row = {
            "head_name": hname,
            "historical_taxonomy_role": hist,
            "role_tol": ROLE_NEAR_ZERO,
        }
        roles = {}
        for ptype in ("BABA", "ABBA"):
            sub = hsub[hsub["pattern_type"] == ptype]
            intact = float(sub["intact_effect"].mean())
            cond = float(sub["conditional_effect"].mean())
            gain = float(sub["conditional_gain"].mean())
            role = classify_role(cond, gain)
            row[f"{ptype}_intact_effect"] = intact
            row[f"{ptype}_conditional_effect"] = cond
            row[f"{ptype}_conditional_gain"] = gain
            row[f"{ptype}_functional_role"] = role
            roles[ptype] = role
        row["role_changes_BABA_vs_ABBA"] = roles["BABA"] != roles["ABBA"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_role_stability(
    path: Path,
    stratum_roles: pd.DataFrame,
    ba_roles: pd.DataFrame,
    hyp: pd.DataFrame,
) -> None:
    lines = [
        "# Functional role stability (Phase-4C audit)",
        "",
        f"Near-zero / mixed tolerance: |conditional_effect| or |conditional_gain| < {ROLE_NEAR_ZERO}",
        "→ classified `near_zero_or_mixed` (not treated as a meaningful role).",
        "",
        "Historical taxonomy labels (e.g. `negative`) are provenance from Phase-1,",
        "not a claim that the head is negative on every prompt distribution.",
        "",
    ]

    # 1. Stable across ≥24/30 strata
    lines.append("## 1. Heads with stable functional role across ≥24/30 strata")
    lines.append("")
    for hname, sub in stratum_roles.groupby("head_name"):
        # Exclude near_zero from "stable role" majority among meaningful, but count mode including all
        counts = sub["functional_role"].value_counts()
        # Prefer most common non-near_zero if it reaches 24; else report top
        meaningful = counts.drop(labels=["near_zero_or_mixed"], errors="ignore")
        if len(meaningful) and meaningful.iloc[0] >= 24:
            role = meaningful.index[0]
            n = int(meaningful.iloc[0])
            lines.append(f"- **{hname}**: `{role}` in {n}/30 strata")
        else:
            top = counts.index[0]
            lines.append(
                f"- {hname}: no single meaningful role ≥24/30 "
                f"(top=`{top}` {int(counts.iloc[0])}/30)"
            )
    lines.append("")

    # 2. Strong BABA/ABBA role reversals
    lines.append("## 2. Strong BABA/ABBA role reversals")
    lines.append("")
    audit_heads = ["L9H7", "L10H2", "L10H10", "L10H7", "L10H1", "L11H2", "L10H6"]
    for h in audit_heads:
        r = ba_roles[ba_roles["head_name"] == h].iloc[0]
        if r["role_changes_BABA_vs_ABBA"]:
            lines.append(
                f"- **{h}**: BABA=`{r['BABA_functional_role']}` vs "
                f"ABBA=`{r['ABBA_functional_role']}` "
                f"(hist taxonomy=`{r['historical_taxonomy_role']}`)"
            )
        else:
            lines.append(
                f"- {h}: same role `{r['BABA_functional_role']}` on BABA and ABBA "
                f"(hist taxonomy=`{r['historical_taxonomy_role']}`)"
            )
    lines.append("")

    # 3. Old two-template role failed to generalize
    lines.append("## 3. Old two-template role failed to generalize?")
    lines.append("")
    h4 = hyp[hyp["hypothesis_id"] == "H4_conditional_effect"].iloc[0]
    lines.append(
        f"- L10H7 H4 conditional_effect<0: **{h4['all30_status']}** "
        f"(mean={h4['aggregate_mean']:.4f}). "
        "Historically a Negative Name Mover; measured conditional effect on canonical IOI is positive."
    )
    l10h1 = hyp[hyp["hypothesis_id"] == "secondary_L10H1_phi_context"].iloc[0]
    lines.append(
        f"- L10H1 suggestive φ_context>0: **{l10h1['all30_status']}** "
        f"(mean={l10h1['aggregate_mean']:.4f})."
    )
    lines.append("")

    # 4. L9H7 and L10H2 still genuine recruits?
    lines.append("## 4. Are L9H7 and L10H2 still genuine recruits under broad templates?")
    lines.append("")
    for hid in ["H1_phi_context", "H2_conditional_gain", "H2_phi_pattern"]:
        r = hyp[hyp["hypothesis_id"] == hid].iloc[0]
        lines.append(f"- {hid}: {r['all30_status']} (mean={r['aggregate_mean']:.4f})")
    for h in ["L9H7", "L10H2"]:
        sub = stratum_roles[stratum_roles["head_name"] == h]
        n_rec = int((sub["functional_role"] == "supportive_recruited").sum())
        lines.append(f"- {h} strata classified supportive_recruited: {n_rec}/30")
    lines.append("")

    # 5. L10H10 description
    lines.append("## 5. L10H10 canonical role description")
    lines.append("")
    sub = stratum_roles[stratum_roles["head_name"] == "L10H10"]
    lines.append(sub["functional_role"].value_counts().to_string())
    lines.append("")
    lines.append(
        "Measured pattern: φ_P>0, φ_C<0, gain<0 (H3 strongly supported). "
        "This is better described as **positive pattern adaptation overwhelmed by context** "
        "rather than merely 'helpful backup downregulated', because the head's pattern "
        "contribution remains positive while context dominates the net conditional gain."
    )
    lines.append("")

    # 6. L10H7 template-dependent?
    lines.append("## 6. Does L10H7 demonstrate template-dependent functional role?")
    lines.append("")
    sub = stratum_roles[stratum_roles["head_name"] == "L10H7"]
    lines.append(sub["functional_role"].value_counts().to_string())
    lines.append("")
    r = ba_roles[ba_roles["head_name"] == "L10H7"].iloc[0]
    lines.append(
        f"- BABA role=`{r['BABA_functional_role']}` "
        f"(cond={r['BABA_conditional_effect']:.3f}, gain={r['BABA_conditional_gain']:.3f})"
    )
    lines.append(
        f"- ABBA role=`{r['ABBA_functional_role']}` "
        f"(cond={r['ABBA_conditional_effect']:.3f}, gain={r['ABBA_conditional_gain']:.3f})"
    )
    lines.append(
        f"- Historical taxonomy=`{r['historical_taxonomy_role']}` (Negative Name Mover provenance); "
        "measured canonical conditional_effect is positive on aggregate — report both without contradiction."
    )
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main():
    assert AGG.exists(), AGG
    print("Backing up pre-fix aggregate artifacts...", flush=True)
    backup_pre_fix()
    old_hyp = pd.read_csv(PRE / "hypothesis_evaluation.csv")

    print("Loading existing raw factorial (no model)...", flush=True)
    df = agg.load_all(FULL)
    df29 = agg.filter_nonoverlap(df)
    st = pd.read_csv(AGG / "stratum_summary.csv")
    st29 = pd.read_csv(AGG / "stratum_summary_nonoverlap29.csv")
    ba = pd.read_csv(AGG / "baba_abba_summary.csv")

    hier, hier_29 = recompute_hierarchical(df, df29)

    print("Hypothesis evaluation...", flush=True)
    hyp = agg.build_hypothesis_evaluation(df, hier, st, st29, hier_29)
    hyp.to_csv(AGG / "hypothesis_evaluation.csv", index=False)

    verdict = agg.scientific_verdict(hyp)
    (AGG / "scientific_verdict.md").write_text(
        f"# Scientific verdict (Phase-4C corrected hierarchical CI)\n\n"
        f"**{verdict}**\n\n"
        "Rule unchanged from Phase-4B preregistration.\n"
        "Secondary hypotheses do not determine the main verdict.\n"
    )
    print(f"VERDICT: {verdict}", flush=True)

    print("Bootstrap fix comparison...", flush=True)
    cmp = compare_bootstrap(old_hyp, hyp)
    cmp.to_csv(AGG / "bootstrap_fix_comparison.csv", index=False)
    assert float(cmp["mean_abs_diff"].max()) < 1e-9, cmp["mean_abs_diff"].max()

    print("Regenerating head reports...", flush=True)
    agg.write_head_report(
        AGG / "L9H7_broad_template_report.md",
        "L9H7",
        "L9H7 broad-template report (Phase-4C CI fix)",
        hyp,
        hier,
        st,
        ba,
        framing="primary H1 — context recruitment",
    )
    agg.write_head_report(
        AGG / "L10H2_broad_template_report.md",
        "L10H2",
        "L10H2 broad-template report (Phase-4C CI fix)",
        hyp,
        hier,
        st,
        ba,
        framing="primary H2 — pattern recruitment (+ secondary S2)",
    )
    agg.write_head_report(
        AGG / "L10H10_broad_template_report.md",
        "L10H10",
        "L10H10 broad-template report (Phase-4C CI fix)",
        hyp,
        hier,
        st,
        ba,
        framing="primary H3 — cancellation (+ secondary S1)",
    )
    agg.write_head_report(
        AGG / "L10H7_broad_template_report.md",
        "L10H7",
        "L10H7 broad-template report (Phase-4C CI fix)",
        hyp,
        hier,
        st,
        ba,
        framing="primary H4 — adaptive-but-suppressive",
    )
    note = (
        "\n## Taxonomy vs measured role\n\n"
        "Historical Phase-1 taxonomy may label this head a Negative Name Mover; "
        "measured conditional_effect on the canonical-template / public-implementation "
        "lexical distribution is positive on aggregate. Both facts are reported without contradiction.\n"
    )
    with open(AGG / "L10H7_broad_template_report.md", "a") as f:
        f.write(note)
    agg.write_head_report(
        AGG / "L10H1_broad_template_report.md",
        "L10H1",
        "L10H1 broad-template report (Phase-4C CI fix)",
        hyp,
        hier,
        st,
        ba,
        framing="secondary / suggestive",
    )

    print("Functional-role audits...", flush=True)
    fr = functional_role_by_stratum(df)
    fr.to_csv(AGG / "functional_role_by_stratum.csv", index=False)
    ba_roles = baba_abba_functional_roles(df)
    ba_roles.to_csv(AGG / "baba_abba_functional_roles.csv", index=False)
    write_role_stability(AGG / "functional_role_stability.md", fr, ba_roles, hyp)

    (AGG / "phase4c_recompute_config.json").write_text(
        json.dumps(
            {
                "phase": "4C",
                "model_execution": False,
                "raw_untouched": True,
                "role_near_zero_tol": ROLE_NEAR_ZERO,
                "verdict": verdict,
                "n_bootstrap": agg.N_BOOT,
                "pre_fix_dir": str(PRE.relative_to(REPO)),
            },
            indent=2,
        )
    )
    print("Done.", flush=True)
    print(cmp[["hypothesis_id", "CI_width_change", "status_changed"]].to_string(index=False))


if __name__ == "__main__":
    main()
