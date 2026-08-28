"""
EF-5 ablation robustness analysis: matched-mean / matched-resample vs zero reference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from extended_fali_ef4_analysis import (
    ARCH_HEADS,
    BROAD_STRATA_MIN,
    DESCRIPTIVE_METRICS,
    EF4_SEEDS,
    MAG_PHI_SCORE_FLOOR,
    N_BOOTSTRAP,
    N_PER_SEED,
    N_STRATA,
    N_TEMPLATES,
    RO_MAG_FRAC,
    TARGET_HEADS,
    baba_abba_heterogeneity,
    ci_direction_ok,
    direction_ok,
    eval_architectural_controls,
    filter_exclude_template_zero,
    hierarchical_ci,
    leave_one_template_out,
    observed_canonical_balanced_mean,
    seed_canonical_means,
    stratum_means_table,
    template_cluster_means,
)
from extended_fali_gpt2 import snapshot_paths, verify_historical_hash_integrity

EF5_SCHEMES = ("zero", "matched_mean", "matched_resample")
ALT_SCHEMES = ("matched_mean", "matched_resample")
INFORMATIVE_FRAC = 0.25
RATIO_DENOM_FLOOR = 0.05

ROBUSTNESS_CHANNELS = (
    {"id": "R1_L10H2_phi_T", "head": "L10H2", "metric": "phi_T", "direction": "positive", "magnitude": False},
    {"id": "R2_L10H2_phi_R", "head": "L10H2", "metric": "phi_R", "direction": "negative", "magnitude": True},
    {"id": "R3_L10H10_phi_T", "head": "L10H10", "metric": "phi_T", "direction": "positive", "magnitude": False},
    {"id": "R4_L10H10_phi_R", "head": "L10H10", "metric": "phi_R", "direction": "negative", "magnitude": True},
)


def bootstrap_rng_seed(scheme: str, head: str, metric: str, scope: str = "primary") -> int:
    from revision_stats import stable_sha256_seed

    return stable_sha256_seed("EF-5", scheme, head, metric, scope)


def hierarchical_ci_scheme(
    values: np.ndarray,
    seed_ids: np.ndarray,
    tid: np.ndarray,
    pattern: np.ndarray,
    *,
    scheme: str,
    head: str,
    metric: str,
    scope: str,
    n_templates: int = N_TEMPLATES,
) -> Tuple[float, float]:
    from revision_stats import hierarchical_template_cluster_bootstrap_means

    boot = hierarchical_template_cluster_bootstrap_means(
        values,
        seed_ids,
        tid,
        pattern,
        n_bootstrap=N_BOOTSTRAP,
        seed=bootstrap_rng_seed(scheme, head, metric, scope),
        n_templates=n_templates,
    )
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def opposite_direction(direction: str) -> str:
    return "negative" if direction == "positive" else "positive"


def broad_confirmed(
    df: pd.DataFrame,
    head: str,
    metric: str,
    direction: str,
    *,
    scheme: str,
    magnitude: bool,
) -> Dict:
    sub = df[df["head_name"] == head]
    values = sub[metric].values
    seeds = sub["seed"].values
    tid = sub["low_level_template_id"].values
    pat = sub["pattern_type"].values

    mean = observed_canonical_balanced_mean(values, seeds, tid, pat)
    lo, hi = hierarchical_ci_scheme(values, seeds, tid, pat, scheme=scheme, head=head, metric=metric, scope="primary")
    seed_means = seed_canonical_means(values, seeds, tid, pat)
    strata = stratum_means_table(df, head, metric)
    n_strata_ok = int(sum(direction_ok(float(r["mean"]), direction) for _, r in strata.iterrows()))

    sub28 = filter_exclude_template_zero(sub)
    v28 = sub28[metric].values
    s28 = sub28["seed"].values
    t28 = sub28["low_level_template_id"].values
    p28 = sub28["pattern_type"].values
    template_ids_14 = list(range(14))
    lo28, hi28 = hierarchical_ci_scheme(
        v28, s28, t28, p28, scheme=scheme, head=head, metric=metric, scope="strict_nonoverlap28", n_templates=14
    )
    seed28 = seed_canonical_means(v28, s28, t28, p28, n_templates=14, template_ids=template_ids_14)

    broad = (
        ci_direction_ok(lo, hi, direction)
        and all(direction_ok(v, direction) for v in seed_means.values())
        and n_strata_ok >= BROAD_STRATA_MIN
    )
    strict = ci_direction_ok(lo28, hi28, direction) and all(direction_ok(v, direction) for v in seed28.values())

    magnitude_pass = True
    magnitude_ratio = None
    if magnitude:
        phi_score_mean = observed_canonical_balanced_mean(sub["Phi_SCORE"].values, seeds, tid, pat)
        if abs(phi_score_mean) < MAG_PHI_SCORE_FLOOR:
            magnitude_pass = False
        else:
            magnitude_pass = abs(mean) >= RO_MAG_FRAC * abs(phi_score_mean)
            magnitude_ratio = abs(mean) / abs(phi_score_mean)

    confirmed = broad and strict and (magnitude_pass if magnitude else True)
    return {
        "exact_mean": mean,
        "ci_low": lo,
        "ci_high": hi,
        "seed_means": seed_means,
        "n_strata_correct": n_strata_ok,
        "n_strata_total": N_STRATA,
        "STRICT_NONOVERLAP_PASS": strict,
        "BROAD_CONFIRMED": broad,
        "magnitude_pass": magnitude_pass if magnitude else None,
        "magnitude_ratio": magnitude_ratio,
        "FINAL_CONFIRMED": confirmed,
    }


def channel_status(
    df: pd.DataFrame,
    channel: Dict,
    *,
    scheme: str,
) -> Dict:
    same = broad_confirmed(
        df,
        channel["head"],
        channel["metric"],
        channel["direction"],
        scheme=scheme,
        magnitude=channel["magnitude"],
    )
    opp_dir = opposite_direction(channel["direction"])
    reversed_eval = broad_confirmed(
        df,
        channel["head"],
        channel["metric"],
        opp_dir,
        scheme=scheme,
        magnitude=False,
    )
    if same["FINAL_CONFIRMED"]:
        status = "ROBUST_SAME"
    elif reversed_eval["BROAD_CONFIRMED"]:
        status = "REVERSED"
    else:
        status = "WEAK_OR_NULL"

    return {
        "channel_id": channel["id"],
        "head": channel["head"],
        "metric": channel["metric"],
        "frozen_direction": channel["direction"],
        "status": status,
        **same,
        "reversed_broad": reversed_eval["BROAD_CONFIRMED"],
    }


def evaluate_damage_strength(damage_df: pd.DataFrame, scheme: str) -> Dict:
    sub = damage_df[damage_df["damage_scheme"] == scheme]
    values = sub["damage_drop"].values
    seeds = sub["seed"].values
    tid = sub["low_level_template_id"].values
    pat = sub["pattern_type"].values
    mean = observed_canonical_balanced_mean(values, seeds, tid, pat)
    lo, hi = hierarchical_ci_scheme(
        values, seeds, tid, pat, scheme=scheme, head="PRIMARY", metric="damage_drop", scope="damage"
    )
    seed_means = seed_canonical_means(values, seeds, tid, pat)
    informative = ci_direction_ok(lo, hi, "positive") and all(v > 0 for v in seed_means.values())
    return {
        "scheme": scheme,
        "exact_mean": mean,
        "ci_low": lo,
        "ci_high": hi,
        "seed_means": seed_means,
        "informative_ci_positive": ci_direction_ok(lo, hi, "positive"),
        "informative_seeds_positive": all(v > 0 for v in seed_means.values()),
        "INFORMATIVE": informative,
    }


def apply_informative_fraction(
    alt_stats: Dict,
    zero_mean: float,
) -> Dict:
    threshold = INFORMATIVE_FRAC * zero_mean
    alt_stats = dict(alt_stats)
    alt_stats["zero_reference_mean"] = zero_mean
    alt_stats["informative_fraction_threshold"] = threshold
    alt_stats["informative_magnitude_pass"] = alt_stats["exact_mean"] >= threshold
    alt_stats["INFORMATIVE"] = (
        alt_stats["INFORMATIVE"] and alt_stats["informative_magnitude_pass"]
    )
    if not alt_stats["informative_magnitude_pass"]:
        alt_stats["informative_label"] = "UNINFORMATIVE"
    elif alt_stats["INFORMATIVE"]:
        alt_stats["informative_label"] = "INFORMATIVE"
    else:
        alt_stats["informative_label"] = "UNINFORMATIVE"
    return alt_stats


def scheme_status(
    channel_results: Dict[str, Dict],
    informative: bool,
) -> str:
    statuses = [c["status"] for c in channel_results.values()]
    if any(s == "REVERSED" for s in statuses) and informative:
        return "FAILED"
    if informative and all(s == "ROBUST_SAME" for s in statuses):
        return "ROBUST"
    if any(s == "REVERSED" for s in statuses):
        return "PARTIAL"  # uninformative reversal not counted as FAILED
    return "PARTIAL"


def ef5_ablation_gate(
    scheme_statuses: Dict[str, str],
    channel_by_scheme: Dict[str, Dict[str, Dict]],
    informative_by_scheme: Dict[str, bool],
) -> Dict:
    mm = scheme_statuses.get("matched_mean")
    mr = scheme_statuses.get("matched_resample")

    reversed_informative = False
    for scheme in ALT_SCHEMES:
        if informative_by_scheme.get(scheme, False):
            for ch in channel_by_scheme[scheme].values():
                if ch["status"] == "REVERSED":
                    reversed_informative = True

    both_informative = all(informative_by_scheme.get(s, False) for s in ALT_SCHEMES)
    dual_fail = False
    if both_informative:
        for ch_def in ROBUSTNESS_CHANNELS:
            cid = ch_def["id"]
            if (
                channel_by_scheme["matched_mean"][cid]["status"] != "ROBUST_SAME"
                and channel_by_scheme["matched_resample"][cid]["status"] != "ROBUST_SAME"
            ):
                dual_fail = True

    if reversed_informative or dual_fail:
        gate = "FAIL"
    elif mm == "ROBUST" and mr == "ROBUST":
        gate = "PASS"
    else:
        gate = "PARTIAL"

    return {
        "EF5_ABLATION_GATE": gate,
        "matched_mean_status": mm,
        "matched_resample_status": mr,
        "informative_reversed_channel": reversed_informative,
        "dual_channel_fail_both_schemes": dual_fail,
    }


def load_scheme_attributions(repo: Path, scheme: str) -> pd.DataFrame:
    parts = []
    for seed in EF4_SEEDS:
        p = repo / f"results/extended_fali/ef5_ablation_robustness/{scheme}/seed_{seed}/attribution_raw.csv"
        parts.append(pd.read_csv(p))
    df = pd.concat(parts, ignore_index=True)
    df["damage_scheme"] = scheme
    return df


def load_zero_reference_attributions(repo: Path) -> pd.DataFrame:
    parts = []
    for seed in EF4_SEEDS:
        p = repo / f"results/extended_fali/ef4_canonical/seed_{seed}/attribution_raw.csv"
        parts.append(pd.read_csv(p))
    df = pd.concat(parts, ignore_index=True)
    df["damage_scheme"] = "zero"
    return df


def load_damage_strength(repo: Path) -> pd.DataFrame:
    parts = []
    for scheme in EF5_SCHEMES:
        if scheme == "zero":
            for seed in EF4_SEEDS:
                p = repo / f"results/extended_fali/ef5_ablation_robustness/zero_damage/seed_{seed}/damage_drop.csv"
                parts.append(pd.read_csv(p))
        else:
            for seed in EF4_SEEDS:
                attr = pd.read_csv(
                    repo / f"results/extended_fali/ef5_ablation_robustness/{scheme}/seed_{seed}/attribution_raw.csv"
                )
                sub = attr[["seed", "example_idx", "damage_drop", "low_level_template_id", "pattern_type", "template_stratum"]].drop_duplicates(
                    subset=["seed", "example_idx"]
                )
                sub["damage_scheme"] = scheme
                parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def effect_size_comparison(
    zero_df: pd.DataFrame,
    mean_df: pd.DataFrame,
    resample_df: pd.DataFrame,
    damage_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    metrics = list(DESCRIPTIVE_METRICS) + ["phi_T", "phi_R"]
    for head in TARGET_HEADS:
        for metric in metrics:
            row = {"head_name": head, "metric": metric}
            for label, df in (("zero", zero_df), ("matched_mean", mean_df), ("matched_resample", resample_df)):
                sub = df[df["head_name"] == head]
                vals = sub[metric].values
                seeds = sub["seed"].values
                tid = sub["low_level_template_id"].values
                pat = sub["pattern_type"].values
                m = observed_canonical_balanced_mean(vals, seeds, tid, pat)
                row[f"{label}_mean"] = m
            z = row.get("zero_mean", 0.0)
            for alt in ("matched_mean", "matched_resample"):
                a = row.get(f"{alt}_mean", 0.0)
                if metric in ("phi_T", "phi_R") and abs(z) >= RATIO_DENOM_FLOOR:
                    row[f"{alt}_over_zero_ratio"] = a / z
                else:
                    row[f"{alt}_over_zero_ratio"] = None
            rows.append(row)

    row = {"head_name": "PRIMARY", "metric": "damage_drop"}
    for label in ("zero", "matched_mean", "matched_resample"):
        sub = damage_df[damage_df["damage_scheme"] == label]
        vals = sub["damage_drop"].values
        seeds = sub["seed"].values
        tid = sub["low_level_template_id"].values
        pat = sub["pattern_type"].values
        row[f"{label}_mean"] = observed_canonical_balanced_mean(vals, seeds, tid, pat)
    row["matched_mean_over_zero_ratio"] = None
    row["matched_resample_over_zero_ratio"] = None
    rows.append(row)
    return pd.DataFrame(rows)


def verify_historical_integrity(repo: Path) -> Dict:
    paths: List[str] = []
    for seed in EF4_SEEDS:
        paths.append(f"results/revision_phase4/full/seed_{seed}/factorial_raw.csv")
        paths.append(f"results/extended_fali/ef4_canonical/seed_{seed}/attribution_raw.csv")
    paths.extend(
        [
            "results/extended_fali/ef4_canonical/aggregate/hypothesis_results.json",
            "results/extended_fali/ef4_canonical/aggregate/fine_score_gate.json",
        ]
    )
    snap = snapshot_paths(repo, paths)
    ok, checks = verify_historical_hash_integrity(snap, snap, paths)
    return {"integrity_pass": ok, "checks": checks, "paths": paths}


def run_analysis(repo: Path, aggregate_dir: Path, *, preregistration_sha: str) -> Dict:
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    integrity = verify_historical_integrity(repo)
    zero_df = load_zero_reference_attributions(repo)
    mean_df = load_scheme_attributions(repo, "matched_mean")
    resample_df = load_scheme_attributions(repo, "matched_resample")
    damage_df = load_damage_strength(repo)
    damage_df.to_csv(aggregate_dir / "damage_strength.csv", index=False)

    zero_strength = evaluate_damage_strength(damage_df, "zero")
    zero_reference_informative = bool(zero_strength["INFORMATIVE"])
    reference_perturbation_stop = not zero_reference_informative

    damage_rows = [zero_strength]
    informative_by_scheme = {"zero": zero_reference_informative}
    for scheme in ALT_SCHEMES:
        st = evaluate_damage_strength(damage_df, scheme)
        st = apply_informative_fraction(st, zero_strength["exact_mean"])
        damage_rows.append(st)
        informative_by_scheme[scheme] = bool(st["INFORMATIVE"])

    channel_by_scheme: Dict[str, Dict[str, Dict]] = {}
    scheme_statuses: Dict[str, str] = {}
    all_channels: Dict[str, Dict] = {}

    for scheme in ALT_SCHEMES:
        df = mean_df if scheme == "matched_mean" else resample_df
        ch_results = {}
        for ch in ROBUSTNESS_CHANNELS:
            res = channel_status(df, ch, scheme=scheme)
            ch_results[ch["id"]] = res
        channel_by_scheme[scheme] = ch_results
        scheme_statuses[scheme] = scheme_status(ch_results, informative_by_scheme[scheme])
        all_channels[scheme] = ch_results

    gate = ef5_ablation_gate(scheme_statuses, channel_by_scheme, informative_by_scheme)
    if reference_perturbation_stop:
        gate["EF5_ABLATION_GATE"] = "FAIL"
        gate["reference_perturbation_stop"] = True
        gate["stop_reason"] = (
            "Zero damage_drop hierarchical CI not entirely > 0; "
            "reference perturbation assumption failed per preregistration."
        )
    else:
        gate["reference_perturbation_stop"] = False

    # exact means / bootstrap for all schemes on target heads
    exact_rows = []
    boot_rows = []
    seed_rows = []
    stratum_parts = []
    for scheme, df in (("zero", zero_df), ("matched_mean", mean_df), ("matched_resample", resample_df)):
        for head in TARGET_HEADS:
            sub = df[df["head_name"] == head]
            for metric in list(DESCRIPTIVE_METRICS) + ["phi_T", "phi_R"]:
                if metric not in sub.columns:
                    continue
                values = sub[metric].values
                seeds = sub["seed"].values
                tid = sub["low_level_template_id"].values
                pat = sub["pattern_type"].values
                mean = observed_canonical_balanced_mean(values, seeds, tid, pat)
                lo, hi = hierarchical_ci_scheme(
                    values, seeds, tid, pat, scheme=scheme, head=head, metric=metric, scope="descriptive"
                )
                exact_rows.append({"damage_scheme": scheme, "head_name": head, "metric": metric, "exact_mean": mean})
                boot_rows.append(
                    {
                        "damage_scheme": scheme,
                        "head_name": head,
                        "metric": metric,
                        "exact_mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                    }
                )
                for s in EF4_SEEDS:
                    ss = sub[sub["seed"] == s]
                    seed_rows.append(
                        {
                            "damage_scheme": scheme,
                            "seed": s,
                            "head_name": head,
                            "metric": metric,
                            "mean": float(ss[metric].mean()),
                        }
                    )
                stratum_parts.append(stratum_means_table(df, head, metric).assign(damage_scheme=scheme))

    pd.DataFrame(exact_rows).to_csv(aggregate_dir / "exact_means.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(aggregate_dir / "hierarchical_bootstrap_ci.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(aggregate_dir / "seed_means.csv", index=False)
    stratum_df = pd.concat(stratum_parts, ignore_index=True)
    stratum_df.to_csv(aggregate_dir / "stratum_means.csv", index=False)

    strict_rows = []
    for scheme in ALT_SCHEMES:
        df = mean_df if scheme == "matched_mean" else resample_df
        for ch in ROBUSTNESS_CHANNELS:
            res = channel_by_scheme[scheme][ch["id"]]
            strict_rows.append(
                {
                    "damage_scheme": scheme,
                    "channel_id": ch["id"],
                    "STRICT_NONOVERLAP_PASS": res["STRICT_NONOVERLAP_PASS"],
                    "FINAL_CONFIRMED": res["FINAL_CONFIRMED"],
                    "status": res["status"],
                }
            )
    pd.DataFrame(strict_rows).to_csv(aggregate_dir / "strict_nonoverlap28_results.csv", index=False)

    effect_df = effect_size_comparison(zero_df, mean_df, resample_df, damage_df)
    effect_df.to_csv(aggregate_dir / "effect_size_comparison.csv", index=False)

    arch_mean = eval_architectural_controls(mean_df)
    arch_resample = eval_architectural_controls(resample_df)

    hetero = []
    for scheme, df in (("matched_mean", mean_df), ("matched_resample", resample_df)):
        for head in TARGET_HEADS:
            for metric in ("phi_T", "phi_R"):
                hetero.append({**baba_abba_heterogeneity(df, head, metric), "damage_scheme": scheme})

    loo_parts = []
    for scheme in ALT_SCHEMES:
        df = mean_df if scheme == "matched_mean" else resample_df
        for ch in ROBUSTNESS_CHANNELS:
            loo = leave_one_template_out(df, ch["head"], ch["metric"], ch["direction"])
            loo["damage_scheme"] = scheme
            loo_parts.append(loo)

    if loo_parts:
        pd.concat(loo_parts, ignore_index=True).to_csv(
            aggregate_dir / "leave_one_template_out.csv", index=False
        )

    (aggregate_dir / "channel_robustness.json").write_text(
        json.dumps({"by_scheme": all_channels, "damage_strength": damage_rows}, indent=2) + "\n"
    )
    (aggregate_dir / "scheme_status.json").write_text(json.dumps(scheme_statuses, indent=2) + "\n")
    (aggregate_dir / "ablation_gate.json").write_text(json.dumps(gate, indent=2) + "\n")

    summary = [
        "# EF-5 ablation robustness summary",
        "",
        f"Preregistration SHA: `{preregistration_sha}`",
        "",
        f"**EF5_ABLATION_GATE: {gate['EF5_ABLATION_GATE']}**",
        "",
        f"- matched_mean: {scheme_statuses['matched_mean']}",
        f"- matched_resample: {scheme_statuses['matched_resample']}",
    ]
    (aggregate_dir / "scientific_summary.md").write_text("\n".join(summary) + "\n")

    analysis_config = {
        "phase": "EF-5",
        "preregistration_sha": preregistration_sha,
        "schemes": list(EF5_SCHEMES),
        "informative_fraction": INFORMATIVE_FRAC,
        "point_estimate": "observed_canonical_balanced_mean",
        "zero_reference": "EF-4 attributions + forward-only zero damage_drop",
    }
    (aggregate_dir / "analysis_config.json").write_text(json.dumps(analysis_config, indent=2) + "\n")

    return {
        "gate": gate,
        "scheme_statuses": scheme_statuses,
        "channel_by_scheme": channel_by_scheme,
        "damage_strength": damage_rows,
        "architectural": {"matched_mean": arch_mean, "matched_resample": arch_resample},
        "heterogeneity": hetero,
        "integrity": integrity,
        "reference_perturbation_stop": reference_perturbation_stop,
    }
