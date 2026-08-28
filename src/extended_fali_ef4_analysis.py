"""
EF-4 canonical-template fine-score confirmation analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from revision_stats import hierarchical_template_cluster_bootstrap_means, stable_sha256_seed

EF4_SEEDS: Tuple[int, ...] = (42, 1, 8, 22)
EF4_HEADS: Tuple[str, ...] = ("L9H7", "L9H0", "L10H2", "L10H10")
TARGET_HEADS: Tuple[str, ...] = ("L10H2", "L10H10")
ARCH_HEADS: Tuple[str, ...] = ("L9H7", "L9H0")
N_PER_SEED = 900
N_STRATA = 30
N_TEMPLATES = 15
N_BOOTSTRAP = 10_000
RO_MAG_FRAC = 0.10
BROAD_STRATA_MIN = 20
STRONG_STRATA_MIN = 24
ARCH_TOL = 1e-6
MAG_PHI_SCORE_FLOOR = 1e-6

PRIMARY_HYPOTHESES = (
    {"id": "H1_L10H2_phi_T_positive", "head": "L10H2", "metric": "phi_T", "direction": "positive", "magnitude": False},
    {"id": "H2_L10H2_phi_R_negative", "head": "L10H2", "metric": "phi_R", "direction": "negative", "magnitude": True},
    {"id": "H3_L10H10_phi_T_positive", "head": "L10H10", "metric": "phi_T", "direction": "positive", "magnitude": False},
    {"id": "H4_L10H10_phi_R_negative", "head": "L10H10", "metric": "phi_R", "direction": "negative", "magnitude": True},
)

DESCRIPTIVE_METRICS = (
    "conditional_gain",
    "Phi_SCORE",
    "phi_V",
    "phi_C",
    "phi_O",
    "desc_phi_target",
    "desc_phi_competing",
    "delta_target_attn",
)


def bootstrap_rng_seed(head: str, metric: str, scope: str = "primary") -> int:
    return stable_sha256_seed("EF-4", head, metric, scope)


def observed_canonical_balanced_mean(
    values: np.ndarray,
    seed_ids: np.ndarray,
    low_level_template_id: np.ndarray,
    pattern_type: np.ndarray,
    *,
    n_templates: int = N_TEMPLATES,
    template_ids: Optional[Sequence[int]] = None,
) -> float:
    """
    Per seed: equal mean over stratum means; then equal mean across seeds.
    """
    values = np.asarray(values, dtype=np.float64)
    seed_ids = np.asarray(seed_ids)
    tid = np.asarray(low_level_template_id).astype(np.int64)
    pattern = np.asarray(pattern_type)
    templates = list(template_ids) if template_ids is not None else list(range(n_templates))
    patterns = ("BABA", "ABBA")
    seed_means: List[float] = []
    for s in EF4_SEEDS:
        mask_s = seed_ids == s
        stratum_means: List[float] = []
        for t in templates:
            for pat in patterns:
                m = mask_s & (tid == t) & (pattern == pat)
                if not np.any(m):
                    continue
                stratum_means.append(float(values[m].mean()))
        if not stratum_means:
            raise ValueError(f"No strata for seed {s}")
        seed_means.append(float(np.mean(stratum_means)))
    return float(np.mean(seed_means))


def seed_canonical_means(
    values: np.ndarray,
    seed_ids: np.ndarray,
    low_level_template_id: np.ndarray,
    pattern_type: np.ndarray,
    *,
    n_templates: int = N_TEMPLATES,
    template_ids: Optional[Sequence[int]] = None,
) -> Dict[int, float]:
    templates = list(template_ids) if template_ids is not None else list(range(n_templates))
    patterns = ("BABA", "ABBA")
    out: Dict[int, float] = {}
    for s in EF4_SEEDS:
        mask_s = seed_ids == s
        if not np.any(mask_s):
            continue
        stratum_means: List[float] = []
        for t in templates:
            for pat in patterns:
                m = mask_s & (low_level_template_id == t) & (pattern_type == pat)
                if np.any(m):
                    stratum_means.append(float(values[m].mean()))
        out[int(s)] = float(np.mean(stratum_means))
    return out


def stratum_means_table(
    df: pd.DataFrame,
    head: str,
    metric: str,
) -> pd.DataFrame:
    sub = df[df["head_name"] == head]
    rows = []
    for (tid, pat), grp in sub.groupby(["low_level_template_id", "pattern_type"]):
        rows.append(
            {
                "head_name": head,
                "metric": metric,
                "low_level_template_id": int(tid),
                "pattern_type": pat,
                "template_stratum": grp["template_stratum"].iloc[0],
                "mean": float(grp[metric].mean()),
                "n": len(grp),
            }
        )
    return pd.DataFrame(rows)


def template_cluster_means(stratum_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (head, metric, tid), grp in stratum_df.groupby(
        ["head_name", "metric", "low_level_template_id"]
    ):
        rows.append(
            {
                "head_name": head,
                "metric": metric,
                "low_level_template_id": int(tid),
                "cluster_mean": float(grp["mean"].mean()),
                "baba_mean": float(grp[grp["pattern_type"] == "BABA"]["mean"].mean())
                if (grp["pattern_type"] == "BABA").any()
                else float("nan"),
                "abba_mean": float(grp[grp["pattern_type"] == "ABBA"]["mean"].mean())
                if (grp["pattern_type"] == "ABBA").any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def hierarchical_ci(
    values: np.ndarray,
    seed_ids: np.ndarray,
    tid: np.ndarray,
    pattern: np.ndarray,
    *,
    head: str,
    metric: str,
    scope: str,
    n_templates: int = N_TEMPLATES,
) -> Tuple[float, float]:
    boot = hierarchical_template_cluster_bootstrap_means(
        values,
        seed_ids,
        tid,
        pattern,
        n_bootstrap=N_BOOTSTRAP,
        seed=bootstrap_rng_seed(head, metric, scope),
        n_templates=n_templates,
    )
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def direction_ok(value: float, direction: str) -> bool:
    if direction == "positive":
        return value > 0
    if direction == "negative":
        return value < 0
    raise ValueError(direction)


def ci_direction_ok(lo: float, hi: float, direction: str) -> bool:
    if direction == "positive":
        return lo > 0 and hi > 0
    if direction == "negative":
        return hi < 0 and lo < 0
    raise ValueError(direction)


def filter_exclude_template_zero(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["low_level_template_id"] != 0].copy()
    out = out.assign(
        low_level_template_id=out["low_level_template_id"].astype(int) - 1
    )
    return out


def evaluate_hypothesis(df: pd.DataFrame, hyp: Dict) -> Dict:
    head = hyp["head"]
    metric = hyp["metric"]
    direction = hyp["direction"]
    sub = df[df["head_name"] == head]
    values = sub[metric].values
    seeds = sub["seed"].values
    tid = sub["low_level_template_id"].values
    pat = sub["pattern_type"].values

    mean = observed_canonical_balanced_mean(values, seeds, tid, pat)
    lo, hi = hierarchical_ci(values, seeds, tid, pat, head=head, metric=metric, scope="primary")
    seed_means = seed_canonical_means(values, seeds, tid, pat)
    strata = stratum_means_table(df, head, metric)
    n_strata_ok = int(sum(direction_ok(float(r["mean"]), direction) for _, r in strata.iterrows()))
    broad = (
        ci_direction_ok(lo, hi, direction)
        and all(direction_ok(v, direction) for v in seed_means.values())
        and n_strata_ok >= BROAD_STRATA_MIN
    )
    strong = broad and n_strata_ok >= STRONG_STRATA_MIN

    sub28 = filter_exclude_template_zero(sub)
    v28 = sub28[metric].values
    s28 = sub28["seed"].values
    t28 = sub28["low_level_template_id"].values
    p28 = sub28["pattern_type"].values
    template_ids_14 = list(range(14))
    mean28 = observed_canonical_balanced_mean(
        v28, s28, t28, p28, n_templates=14, template_ids=template_ids_14
    )
    lo28, hi28 = hierarchical_ci(
        v28, s28, t28, p28, head=head, metric=metric, scope="strict_nonoverlap28", n_templates=14
    )
    seed28 = seed_canonical_means(
        v28, s28, t28, p28, n_templates=14, template_ids=template_ids_14
    )
    strict = ci_direction_ok(lo28, hi28, direction) and all(
        direction_ok(v, direction) for v in seed28.values()
    )

    magnitude_pass = True
    magnitude_ratio = None
    threshold = None
    if hyp["magnitude"]:
        phi_score_mean = observed_canonical_balanced_mean(
            sub["Phi_SCORE"].values, seeds, tid, pat
        )
        if abs(phi_score_mean) < MAG_PHI_SCORE_FLOOR:
            magnitude_pass = False
            threshold = None
        else:
            threshold = RO_MAG_FRAC * abs(phi_score_mean)
            magnitude_pass = abs(mean) >= threshold
            magnitude_ratio = abs(mean) / abs(phi_score_mean)

    if hyp["magnitude"]:
        final_confirmed = broad and strict and magnitude_pass
    else:
        final_confirmed = broad and strict

    return {
        "hypothesis_id": hyp["id"],
        "head": head,
        "metric": metric,
        "direction": direction,
        "exact_mean": mean,
        "ci_low": lo,
        "ci_high": hi,
        "seed_means": seed_means,
        "n_strata_correct": n_strata_ok,
        "n_strata_total": N_STRATA,
        "BROAD_CONFIRMED": broad,
        "STRONG": strong,
        "strict_nonoverlap_mean": mean28,
        "strict_nonoverlap_ci_low": lo28,
        "strict_nonoverlap_ci_high": hi28,
        "strict_nonoverlap_seed_means": seed28,
        "STRICT_NONOVERLAP_PASS": strict,
        "magnitude_threshold": threshold,
        "magnitude_ratio": magnitude_ratio,
        "magnitude_pass": magnitude_pass if hyp["magnitude"] else None,
        "FINAL_CONFIRMED": final_confirmed,
    }


def leave_one_template_out(df: pd.DataFrame, head: str, metric: str, direction: str) -> pd.DataFrame:
    sub = df[df["head_name"] == head]
    rows = []
    for exclude_tid in range(N_TEMPLATES):
        kept = sub[sub["low_level_template_id"] != exclude_tid]
        templates = [t for t in range(N_TEMPLATES) if t != exclude_tid]
        mean = observed_canonical_balanced_mean(
            kept[metric].values,
            kept["seed"].values,
            kept["low_level_template_id"].values,
            kept["pattern_type"].values,
            n_templates=N_TEMPLATES,
            template_ids=templates,
        )
        rows.append(
            {
                "head_name": head,
                "metric": metric,
                "excluded_template_id": exclude_tid,
                "observed_mean": mean,
                "predicted_sign": direction_ok(mean, direction),
            }
        )
    out = pd.DataFrame(rows)
    return out


def eval_architectural_controls(df: pd.DataFrame) -> Dict:
    results = {}
    for head in ARCH_HEADS:
        sub = df[df["head_name"] == head]
        results[head] = {
            "max_abs_phi_T": float(sub["phi_T"].abs().max()),
            "max_abs_phi_R": float(sub["phi_R"].abs().max()),
            "max_abs_phi_O": float(sub["phi_O"].abs().max()),
            "max_abs_phi_V": float(sub["phi_V"].abs().max()),
            "max_context_residual": float((sub["phi_C"] - sub["conditional_gain"]).abs().max()),
            "architectural_pass": bool(
                sub["phi_T"].abs().max() <= ARCH_TOL
                and sub["phi_R"].abs().max() <= ARCH_TOL
                and sub["phi_O"].abs().max() <= ARCH_TOL
                and sub["phi_V"].abs().max() <= ARCH_TOL
                and (sub["phi_C"] - sub["conditional_gain"]).abs().max() <= ARCH_TOL
            ),
        }
    return results


def fine_score_gate(hypothesis_results: Dict) -> Dict:
    def head_passes(head: str) -> bool:
        t_key = f"H{'1' if head == 'L10H2' else '3'}_{head}_phi_T_positive"
        r_key = f"H{'2' if head == 'L10H2' else '4'}_{head}_phi_R_negative"
        t = hypothesis_results[t_key]["FINAL_CONFIRMED"]
        r = hypothesis_results[r_key]["FINAL_CONFIRMED"]
        return bool(t and r)

    h2 = head_passes("L10H2")
    h10 = head_passes("L10H10")
    if h2 and h10:
        decision = "PASS"
    elif h2 or h10:
        decision = "PARTIAL"
    else:
        decision = "FAIL"
    return {
        "EF4_FINE_SCORE_GATE": decision,
        "L10H2_both_channels": h2,
        "L10H10_both_channels": h10,
    }


def pattern_aggregate_mean(
    df: pd.DataFrame,
    head: str,
    metric: str,
    pattern: str,
) -> float:
    sub = df[(df["head_name"] == head) & (df["pattern_type"] == pattern)]
    seed_means: List[float] = []
    for s in EF4_SEEDS:
        ss = sub[sub["seed"] == s]
        tmpl_means = [
            float(ss[ss["low_level_template_id"] == t][metric].mean())
            for t in range(N_TEMPLATES)
        ]
        seed_means.append(float(np.mean(tmpl_means)))
    return float(np.mean(seed_means))


def baba_abba_heterogeneity(df: pd.DataFrame, head: str, metric: str) -> Dict:
    baba = pattern_aggregate_mean(df, head, metric, "BABA")
    abba = pattern_aggregate_mean(df, head, metric, "ABBA")
    opposite = (baba > 0 and abba < 0) or (baba < 0 and abba > 0)
    return {
        "head_name": head,
        "metric": metric,
        "baba_mean": baba,
        "abba_mean": abba,
        "opposite_signs": opposite,
    }


def load_merged_attributions(repo_root: Path) -> pd.DataFrame:
    parts = []
    for seed in EF4_SEEDS:
        p = repo_root / f"results/extended_fali/ef4_canonical/seed_{seed}/attribution_raw.csv"
        parts.append(pd.read_csv(p))
    return pd.concat(parts, ignore_index=True)


def run_analysis(repo_root: Path, aggregate_dir: Path, *, preregistration_sha: str) -> Dict:
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    df = load_merged_attributions(repo_root)
    df.to_csv(aggregate_dir / "attribution_merged_raw.csv", index=False)

    exact_rows = []
    boot_rows = []
    seed_rows = []
    stratum_parts = []
    cluster_parts = []
    loo_parts = []
    hetero_rows = []

    for head in EF4_HEADS:
        sub = df[df["head_name"] == head]
        for metric in list(DESCRIPTIVE_METRICS) + ("phi_T", "phi_R"):
            if metric not in sub.columns:
                continue
            values = sub[metric].values
            seeds = sub["seed"].values
            tid = sub["low_level_template_id"].values
            pat = sub["pattern_type"].values
            mean = observed_canonical_balanced_mean(values, seeds, tid, pat)
            lo, hi = hierarchical_ci(values, seeds, tid, pat, head=head, metric=metric, scope="descriptive")
            exact_rows.append({"head_name": head, "metric": metric, "exact_mean": mean})
            boot_rows.append(
                {
                    "head_name": head,
                    "metric": metric,
                    "exact_mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "seed_means": json.dumps(seed_canonical_means(values, seeds, tid, pat)),
                }
            )
            for s in EF4_SEEDS:
                ss = sub[sub["seed"] == s]
                seed_rows.append(
                    {
                        "seed": s,
                        "head_name": head,
                        "metric": metric,
                        "mean": float(ss[metric].mean()),
                        "n": len(ss),
                    }
                )
            sm = stratum_means_table(df, head, metric)
            stratum_parts.append(sm)

    stratum_df = pd.concat(stratum_parts, ignore_index=True)
    stratum_df.to_csv(aggregate_dir / "stratum_means.csv", index=False)
    cluster_df = template_cluster_means(stratum_df)
    cluster_df.to_csv(aggregate_dir / "template_cluster_means.csv", index=False)
    pd.DataFrame(exact_rows).to_csv(aggregate_dir / "exact_means.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(aggregate_dir / "hierarchical_bootstrap_ci.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(aggregate_dir / "seed_means.csv", index=False)

    hypothesis_results = {}
    strict_rows = []
    for hyp in PRIMARY_HYPOTHESES:
        res = evaluate_hypothesis(df, hyp)
        hypothesis_results[res["hypothesis_id"]] = res
        strict_rows.append(
            {
                "hypothesis_id": res["hypothesis_id"],
                "head": res["head"],
                "metric": res["metric"],
                "strict_nonoverlap_mean": res["strict_nonoverlap_mean"],
                "strict_nonoverlap_ci_low": res["strict_nonoverlap_ci_low"],
                "strict_nonoverlap_ci_high": res["strict_nonoverlap_ci_high"],
                "STRICT_NONOVERLAP_PASS": res["STRICT_NONOVERLAP_PASS"],
            }
        )
        loo_parts.append(leave_one_template_out(df, hyp["head"], hyp["metric"], hyp["direction"]))

    pd.DataFrame(strict_rows).to_csv(aggregate_dir / "strict_nonoverlap28_results.csv", index=False)
    pd.concat(loo_parts, ignore_index=True).to_csv(
        aggregate_dir / "leave_one_template_out.csv", index=False
    )

    arch = eval_architectural_controls(df)
    gate = fine_score_gate(hypothesis_results)

    for head in TARGET_HEADS:
        for metric in ("phi_T", "phi_R", "phi_O"):
            hetero_rows.append(baba_abba_heterogeneity(df, head, metric))

    (aggregate_dir / "hypothesis_results.json").write_text(
        json.dumps({"architectural": arch, "hypotheses": hypothesis_results}, indent=2) + "\n"
    )
    (aggregate_dir / "fine_score_gate.json").write_text(json.dumps(gate, indent=2) + "\n")

    summary = [
        "# EF-4 canonical fine-score confirmation summary",
        "",
        f"Preregistration SHA: `{preregistration_sha}`",
        "",
        f"**EF4_FINE_SCORE_GATE: {gate['EF4_FINE_SCORE_GATE']}**",
        "",
        "## Primary hypotheses",
    ]
    for hid, res in hypothesis_results.items():
        summary.append(
            f"- {hid}: FINAL_CONFIRMED={res['FINAL_CONFIRMED']} "
            f"mean={res['exact_mean']:.6f} CI=[{res['ci_low']:.6f}, {res['ci_high']:.6f}] "
            f"strata={res['n_strata_correct']}/{res['n_strata_total']}"
        )
    summary.append("")
    summary.append("Coarse P/V/C quantities are previously observed on these cohorts; "
                     "new outcomes are fine-grained T/R/O attributions.")
    (aggregate_dir / "scientific_summary.md").write_text("\n".join(summary) + "\n")

    analysis_config = {
        "phase": "EF-4",
        "preregistration_sha": preregistration_sha,
        "seeds": list(EF4_SEEDS),
        "heads": list(EF4_HEADS),
        "n_per_seed": N_PER_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "point_estimate": "observed_canonical_balanced_mean",
        "bootstrap": "hierarchical_template_cluster_bootstrap_means (Phase-4C corrected)",
        "strict_nonoverlap": "exclude template_id=0 entirely (28 strata, 14 clusters)",
    }
    (aggregate_dir / "analysis_config.json").write_text(json.dumps(analysis_config, indent=2) + "\n")

    return {
        "hypothesis_results": hypothesis_results,
        "architectural": arch,
        "fine_score_gate": gate,
        "heterogeneity": hetero_rows,
    }
