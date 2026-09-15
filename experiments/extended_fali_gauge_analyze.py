#!/usr/bin/env python3
"""Paired gauge-sensitivity analysis vs native EF-4 / EF-6 results."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from extended_fali_ef4_analysis import (  # noqa: E402
    EF4_SEEDS,
    N_BOOTSTRAP,
    N_TEMPLATES,
    hierarchical_ci,
    observed_canonical_balanced_mean,
)
from extended_fali_ef6_analysis import (  # noqa: E402
    CONFIRMATION_SEEDS,
    observed_offset_balanced_mean,
    percentile_ci,
)
from revision_stats import hierarchical_template_cluster_bootstrap_means, stable_sha256_seed  # noqa: E402

OUT = _REPO / "results/extended_fali/gauge_sensitivity"
NATIVE_EF4 = _REPO / "results/extended_fali/ef4_canonical"
NATIVE_EF6 = _REPO / "results/extended_fali/ef6_pythia_induction"
GAUGE_EF4 = OUT / "ef4_rowmean"
GAUGE_EF6 = OUT / "ef6_rowmean"

FACTORS = ("phi_T", "phi_R", "phi_O", "phi_V", "phi_C", "Phi_SCORE")
GAIN = "conditional_gain"
HEADS_EF4 = ("L10H2", "L10H10")
HEAD_EF6 = "L8H2"
GAUGE_PREREG = "e60f81acfff8fc9dd4f924ebb1c087ec89734c05"


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_ef4(native: bool) -> pd.DataFrame:
    root = NATIVE_EF4 if native else GAUGE_EF4
    frames = []
    for seed in EF4_SEEDS:
        p = root / f"seed_{seed}" / "attribution_raw.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        df = pd.read_csv(p)
        df = df[df["head_name"].isin(HEADS_EF4)].copy()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_ef6(native: bool) -> pd.DataFrame:
    root = (NATIVE_EF6 / "confirmation") if native else GAUGE_EF6
    frames = []
    for seed in CONFIRMATION_SEEDS:
        p = root / f"seed_{seed}" / "attribution_raw.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    return df[df["head_name"] == HEAD_EF6].copy()


def decide_sign(native_mean: float, row_mean: float, lo: float, hi: float, expected_sign: int) -> str:
    """expected_sign: +1 or -1 for the native/published direction of this factor."""
    if np.sign(row_mean) != np.sign(native_mean) and row_mean != 0 and native_mean != 0:
        return "SIGN-CHANGED"
    # same sign as native
    if expected_sign > 0:
        if lo > 0 and hi > 0 and row_mean > 0:
            return "PRESERVED"
        if row_mean > 0:
            return "SIGN-UNCERTAIN"
        return "SIGN-CHANGED"
    if expected_sign < 0:
        if lo < 0 and hi < 0 and row_mean < 0:
            return "PRESERVED"
        if row_mean < 0:
            return "SIGN-UNCERTAIN"
        return "SIGN-CHANGED"
    return "SIGN-UNCERTAIN"


def paired_ef4_metric(joined: pd.DataFrame, head: str, metric: str) -> Dict:
    sub = joined[joined["head_name"] == head].copy()
    n_vals = sub[f"{metric}_n"].values
    r_vals = sub[f"{metric}_r"].values
    d_vals = r_vals - n_vals
    seeds = sub["seed"].values
    tid = sub["low_level_template_id"].values
    pat = sub["pattern_type"].values

    n_mean = observed_canonical_balanced_mean(n_vals, seeds, tid, pat)
    r_mean = observed_canonical_balanced_mean(r_vals, seeds, tid, pat)
    d_mean = observed_canonical_balanced_mean(d_vals, seeds, tid, pat)
    n_lo, n_hi = hierarchical_ci(n_vals, seeds, tid, pat, head=head, metric=f"native_{metric}", scope="gauge")
    r_lo, r_hi = hierarchical_ci(r_vals, seeds, tid, pat, head=head, metric=f"rowmean_{metric}", scope="gauge")
    # Paired delta CI with dedicated seed
    boot = hierarchical_template_cluster_bootstrap_means(
        d_vals,
        seeds,
        tid,
        pat,
        n_bootstrap=N_BOOTSTRAP,
        seed=stable_sha256_seed("GAUGE", "EF4", head, metric, "paired_delta"),
        n_templates=N_TEMPLATES,
    )
    d_lo, d_hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))

    sign_flip_frac = float(
        np.mean((np.sign(n_vals) != np.sign(r_vals)) & (n_vals != 0) & (r_vals != 0))
    )
    pearson = float(stats.pearsonr(n_vals, r_vals).statistic) if len(n_vals) > 2 else float("nan")
    spearman = float(stats.spearmanr(n_vals, r_vals).statistic) if len(n_vals) > 2 else float("nan")
    abs_n = float(np.mean(np.abs(n_vals)))
    abs_r = float(np.mean(np.abs(r_vals)))
    return {
        "head": head,
        "metric": metric,
        "native_mean": n_mean,
        "native_ci_lo": n_lo,
        "native_ci_hi": n_hi,
        "rowmean_mean": r_mean,
        "rowmean_ci_lo": r_lo,
        "rowmean_ci_hi": r_hi,
        "paired_delta_mean": d_mean,
        "paired_delta_ci_lo": d_lo,
        "paired_delta_ci_hi": d_hi,
        "sign_flip_fraction": sign_flip_frac,
        "pearson": pearson,
        "spearman": spearman,
        "mean_abs_native": abs_n,
        "mean_abs_rowmean": abs_r,
    }


def paired_ef6_metric(joined: pd.DataFrame, metric: str) -> Dict:
    n_vals = joined[f"{metric}_n"].values
    r_vals = joined[f"{metric}_r"].values
    d_vals = r_vals - n_vals
    offsets = joined["query_offset"].values
    # seed×offset balancing: pool all confirmation seeds; use offset-balanced mean
    # For multi-seed: average offset-balanced means across seeds equally
    seed_ids = joined["seed"].values

    def multi_seed_offset_mean(values: np.ndarray) -> float:
        means = []
        for s in CONFIRMATION_SEEDS:
            m = seed_ids == s
            means.append(observed_offset_balanced_mean(values[m], offsets[m]))
        return float(np.mean(means))

    n_mean = multi_seed_offset_mean(n_vals)
    r_mean = multi_seed_offset_mean(r_vals)
    d_mean = multi_seed_offset_mean(d_vals)

    # Bootstrap: resample within seed×offset, average seed means (match confirmation style)
    rng = np.random.default_rng(stable_sha256_seed("GAUGE", "EF6", HEAD_EF6, metric, "paired"))
    boot = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        seed_means = []
        for s in CONFIRMATION_SEEDS:
            m = seed_ids == s
            v = values_draw = d_vals[m]
            off = offsets[m]
            # resample within each offset
            parts = []
            for o in np.unique(off):
                idx = np.where(off == o)[0]
                draw = rng.choice(idx, size=len(idx), replace=True)
                parts.append(values_draw[draw].mean())
            seed_means.append(float(np.mean(parts)))
        boot[b] = float(np.mean(seed_means))
    d_lo, d_hi = percentile_ci(boot)

    # marginal CIs for native/rowmean similarly
    def boot_ci(values: np.ndarray) -> Tuple[float, float]:
        bmeans = np.empty(N_BOOTSTRAP)
        for b in range(N_BOOTSTRAP):
            seed_means = []
            for s in CONFIRMATION_SEEDS:
                m = seed_ids == s
                vv = values[m]
                off = offsets[m]
                parts = []
                for o in np.unique(off):
                    idx = np.where(off == o)[0]
                    draw = rng.choice(idx, size=len(idx), replace=True)
                    parts.append(vv[draw].mean())
                seed_means.append(float(np.mean(parts)))
            bmeans[b] = float(np.mean(seed_means))
        return percentile_ci(bmeans)

    n_lo, n_hi = boot_ci(n_vals)
    r_lo, r_hi = boot_ci(r_vals)
    sign_flip_frac = float(
        np.mean((np.sign(n_vals) != np.sign(r_vals)) & (n_vals != 0) & (r_vals != 0))
    )
    pearson = float(stats.pearsonr(n_vals, r_vals).statistic)
    spearman = float(stats.spearmanr(n_vals, r_vals).statistic)
    return {
        "head": HEAD_EF6,
        "metric": metric,
        "native_mean": n_mean,
        "native_ci_lo": n_lo,
        "native_ci_hi": n_hi,
        "rowmean_mean": r_mean,
        "rowmean_ci_lo": r_lo,
        "rowmean_ci_hi": r_hi,
        "paired_delta_mean": d_mean,
        "paired_delta_ci_lo": d_lo,
        "paired_delta_ci_hi": d_hi,
        "sign_flip_fraction": sign_flip_frac,
        "pearson": pearson,
        "spearman": spearman,
        "mean_abs_native": float(np.mean(np.abs(n_vals))),
        "mean_abs_rowmean": float(np.mean(np.abs(r_vals))),
    }


def factor_decision(row: Dict, expected: Dict[str, int]) -> str:
    metric = row["metric"]
    if metric not in expected:
        return "n/a"
    return decide_sign(
        row["native_mean"],
        row["rowmean_mean"],
        row["rowmean_ci_lo"],
        row["rowmean_ci_hi"],
        expected[metric],
    )


def fmt(x: float) -> str:
    return f"{x:.6g}"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    # Verify prereg SHA recorded
    prereg_ok = True
    for seed in EF4_SEEDS:
        p = GAUGE_EF4 / f"seed_{seed}" / "preregistration_sha.txt"
        if not p.exists() or p.read_text().strip() != GAUGE_PREREG:
            prereg_ok = False
    for seed in CONFIRMATION_SEEDS:
        p = GAUGE_EF6 / f"seed_{seed}" / "preregistration_sha.txt"
        if not p.exists() or p.read_text().strip() != GAUGE_PREREG:
            prereg_ok = False

    native4 = load_ef4(True)
    gauge4 = load_ef4(False)
    keys4 = ["seed", "example_idx", "head_name", "low_level_template_id", "pattern_type"]
    j4 = native4.merge(gauge4, on=keys4, suffixes=("_n", "_r"), how="inner")
    if len(j4) != len(native4) or len(j4) != len(gauge4):
        raise AssertionError(f"EF4 join mismatch: native={len(native4)} gauge={len(gauge4)} joined={len(j4)}")

    native6 = load_ef6(True)
    gauge6 = load_ef6(False)
    keys6 = ["seed", "example_idx", "query_offset", "head_name"]
    j6 = native6.merge(gauge6, on=keys6, suffixes=("_n", "_r"), how="inner")
    if len(j6) != len(native6) or len(j6) != len(gauge6):
        raise AssertionError(f"EF6 join mismatch: native={len(native6)} gauge={len(gauge6)} joined={len(j6)}")

    expected4 = {
        "L10H2": {"phi_T": 1, "phi_R": -1},
        "L10H10": {"phi_T": 1, "phi_R": -1},
    }
    expected6 = {"phi_T": -1, "phi_R": 1}

    rows: List[Dict] = []
    decisions: Dict[str, Dict[str, str]] = {}

    for head in HEADS_EF4:
        decisions[head] = {}
        for metric in list(FACTORS) + [GAIN]:
            rec = paired_ef4_metric(j4, head, metric)
            if metric in ("phi_T", "phi_R"):
                rec["decision"] = factor_decision(rec, expected4[head])
                decisions[head][metric] = rec["decision"]
            else:
                rec["decision"] = "n/a"
            rows.append({**rec, "setting": "GPT2_EF4"})

    decisions[HEAD_EF6] = {}
    for metric in list(FACTORS) + [GAIN]:
        rec = paired_ef6_metric(j6, metric)
        if metric in ("phi_T", "phi_R"):
            rec["decision"] = factor_decision(rec, expected6)
            decisions[HEAD_EF6][metric] = rec["decision"]
        else:
            rec["decision"] = "n/a"
        rows.append({**rec, "setting": "Pythia_EF6"})

    # Cross-setting contrast
    def preserved_head(head: str, t_sign: int, r_sign: int) -> bool:
        # Need PRESERVED on both T and R under rowmean with directional CI
        return decisions[head].get("phi_T") == "PRESERVED" and decisions[head].get("phi_R") == "PRESERVED"

    cross = (
        preserved_head("L10H2", 1, -1)
        and preserved_head("L10H10", 1, -1)
        and preserved_head(HEAD_EF6, -1, 1)
    )
    cross_label = "PRESERVED" if cross else "NOT PRESERVED"

    # Invariants
    inv = {
        "ef4_max_abs_gain_delta": float((j4["conditional_gain_r"] - j4["conditional_gain_n"]).abs().max()),
        "ef4_max_abs_Phi_SCORE_delta": float((j4["Phi_SCORE_r"] - j4["Phi_SCORE_n"]).abs().max()),
        "ef4_max_owen_residual_rowmean": float(j4["owen_residual_r"].max()) if "owen_residual_r" in j4 else None,
        "ef6_max_abs_gain_delta": float((j6["conditional_gain_r"] - j6["conditional_gain_n"]).abs().max()),
        "ef6_max_abs_Phi_SCORE_delta": float((j6["Phi_SCORE_r"] - j6["Phi_SCORE_n"]).abs().max()),
        "ef6_max_owen_residual_rowmean": float(j6["owen_residual_r"].max()) if "owen_residual_r" in j6 else None,
        "preregistration_sha_recorded": prereg_ok,
        "cross_setting_fine_contrast": cross_label,
    }
    # Quotient bridge / efficiency from rowmean attribution
    for label, df in (("ef4", j4), ("ef6", j6)):
        bridge = (
            df["Phi_SCORE_r"] - (df["phi_T_r"] + df["phi_R_r"] + df["phi_O_r"])
        ).abs().max()
        inv[f"{label}_max_quotient_bridge_err"] = float(bridge)
        eff = (
            df["phi_T_r"]
            + df["phi_R_r"]
            + df["phi_O_r"]
            + df["phi_V_r"]
            + df["phi_C_r"]
            - (df["U11111_r"] - df["U00000_r"])
        ).abs().max()
        inv[f"{label}_max_owen_efficiency_err"] = float(eff)

    # Native file integrity
    native_hashes = {}
    for seed in EF4_SEEDS:
        p = NATIVE_EF4 / f"seed_{seed}" / "attribution_raw.csv"
        native_hashes[f"ef4_seed_{seed}"] = file_sha256(p)
    for seed in CONFIRMATION_SEEDS:
        p = NATIVE_EF6 / "confirmation" / f"seed_{seed}" / "attribution_raw.csv"
        native_hashes[f"ef6_seed_{seed}"] = file_sha256(p)
    inv["native_attribution_sha256"] = native_hashes

    exact = pd.DataFrame(rows)
    exact.to_csv(OUT / "gauge_sensitivity_exact_means.csv", index=False)
    deltas = exact[
        [
            "setting",
            "head",
            "metric",
            "paired_delta_mean",
            "paired_delta_ci_lo",
            "paired_delta_ci_hi",
            "sign_flip_fraction",
            "pearson",
            "spearman",
            "decision",
        ]
    ]
    deltas.to_csv(OUT / "gauge_sensitivity_paired_differences.csv", index=False)

    summary = {
        "gauge_preregistration_sha": GAUGE_PREREG,
        "decisions": decisions,
        "cross_setting_fine_contrast": cross_label,
        "invariants": inv,
        "rows": rows,
    }
    (OUT / "gauge_sensitivity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "full_invariant_report.json").write_text(json.dumps(inv, indent=2) + "\n")

    # Interpretation
    if cross:
        interp = "A"
    elif any(
        decisions[h].get(m) == "SIGN-CHANGED"
        for h in ("L10H2", "L10H10", HEAD_EF6)
        for m in ("phi_T", "phi_R")
    ):
        interp = "C"
    else:
        interp = "B"

    # Report
    lines = [
        "# GAUGE SENSITIVITY REPORT",
        "",
        f"Preregistration SHA: `{GAUGE_PREREG}`",
        f"Preregistration SHA recorded in runs: {'YES' if prereg_ok else 'NO'}",
        f"Cross-setting fine contrast: **{cross_label}**",
        f"Scientific interpretation code: **{interp}**",
        "",
    ]

    def section(setting: str, head: str):
        lines.append(f"## {setting} {head}")
        lines.append("")
        lines.append("| metric | native mean [CI] | rowmean mean [CI] | paired Δ [CI] | decision |")
        lines.append("|---|---|---|---|---|")
        for metric in list(FACTORS) + [GAIN]:
            r = exact[(exact["setting"] == setting) & (exact["head"] == head) & (exact["metric"] == metric)].iloc[0]
            lines.append(
                f"| {metric} | {fmt(r['native_mean'])} [{fmt(r['native_ci_lo'])}, {fmt(r['native_ci_hi'])}] "
                f"| {fmt(r['rowmean_mean'])} [{fmt(r['rowmean_ci_lo'])}, {fmt(r['rowmean_ci_hi'])}] "
                f"| {fmt(r['paired_delta_mean'])} [{fmt(r['paired_delta_ci_lo'])}, {fmt(r['paired_delta_ci_hi'])}] "
                f"| {r['decision']} |"
            )
        lines.append("")

    section("GPT2_EF4", "L10H2")
    section("GPT2_EF4", "L10H10")
    section("Pythia_EF6", HEAD_EF6)

    lines.extend(
        [
            "## Invariants",
            "",
            f"- Owen efficiency max err EF4/EF6: {inv['ef4_max_owen_efficiency_err']}, {inv['ef6_max_owen_efficiency_err']}",
            f"- Quotient bridge max err EF4/EF6: {inv['ef4_max_quotient_bridge_err']}, {inv['ef6_max_quotient_bridge_err']}",
            f"- Coarse SCORE max |Δ| EF4/EF6: {inv['ef4_max_abs_Phi_SCORE_delta']}, {inv['ef6_max_abs_Phi_SCORE_delta']}",
            f"- Gain max |Δ| EF4/EF6: {inv['ef4_max_abs_gain_delta']}, {inv['ef6_max_abs_gain_delta']}",
            "",
            "## Claim boundary",
            "",
            "Even if signs survive row-centering, this does **not** establish universal gauge invariance.",
            "Allowed: robustness to the prespecified zero-mean gauge sensitivity only.",
            "",
        ]
    )
    (OUT / "GAUGE_SENSITIVITY_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"cross": cross_label, "interp": interp, "prereg_ok": prereg_ok}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
