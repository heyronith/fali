"""
Phase-4B aggregation: broad-template factorial validation.

Reads results/revision_phase4/full/seed_*/factorial_raw.csv
Writes results/revision_phase4/full/aggregate/

Frozen primary hypotheses H1–H4 and secondary S1/S2 are evaluated with
pre-registered BROADLY_SUPPORTED / STRONGLY_BROADLY_SUPPORTED rules.
Does NOT redefine hypotheses after observing results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from revision_stats import (  # noqa: E402
    hierarchical_template_cluster_bootstrap_ci,
    stable_sha256_seed,
    stratified_bootstrap_ci,
    template_cluster_bootstrap_ci,
    within_template_stratified_bootstrap_ci,
)

SEEDS = [42, 1, 8, 22]
N_BOOT = 10000
N_PER_SEED = 900
N_STRATA = 30
N_PER_STRATUM = 30
EXACT_OVERLAP_STRATUM = "T00_BABA"  # only exact legacy overlap

METRICS = [
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
]

PRIMARY_BOOT_METRICS = [
    "conditional_effect",
    "conditional_gain",
    "phi_pattern",
    "phi_value",
    "phi_context",
    "interaction_PC",
    "intact_effect",
    "main_P",
    "main_V",
    "main_C",
    "interaction_PV",
    "interaction_VC",
    "interaction_PVC",
    "pattern_effect_B_VB",
    "value_effect_B_PB",
    "context_effect_PB_VB",
]

# Architectural near-zero tolerance for L9H7 phi_P / phi_V
ARCH_TOL = 1e-4

# Frozen hypothesis specs (do not edit after results)
HYPOTHESES = [
    {
        "hypothesis_id": "H1_phi_pattern",
        "head_name": "L9H7",
        "metric": "phi_pattern",
        "predicted_direction": "approx_zero",
        "primary": True,
        "architectural": True,
    },
    {
        "hypothesis_id": "H1_phi_value",
        "head_name": "L9H7",
        "metric": "phi_value",
        "predicted_direction": "approx_zero",
        "primary": True,
        "architectural": True,
    },
    {
        "hypothesis_id": "H1_phi_context",
        "head_name": "L9H7",
        "metric": "phi_context",
        "predicted_direction": ">",
        "primary": True,
        "architectural": False,
    },
    {
        "hypothesis_id": "H2_conditional_gain",
        "head_name": "L10H2",
        "metric": "conditional_gain",
        "predicted_direction": ">",
        "primary": True,
        "architectural": False,
    },
    {
        "hypothesis_id": "H2_phi_pattern",
        "head_name": "L10H2",
        "metric": "phi_pattern",
        "predicted_direction": ">",
        "primary": True,
        "architectural": False,
    },
    {
        "hypothesis_id": "H3_phi_pattern",
        "head_name": "L10H10",
        "metric": "phi_pattern",
        "predicted_direction": ">",
        "primary": True,
        "architectural": False,
    },
    {
        "hypothesis_id": "H3_phi_context",
        "head_name": "L10H10",
        "metric": "phi_context",
        "predicted_direction": "<",
        "primary": True,
        "architectural": False,
    },
    {
        "hypothesis_id": "H3_conditional_gain",
        "head_name": "L10H10",
        "metric": "conditional_gain",
        "predicted_direction": "<",
        "primary": True,
        "architectural": False,
    },
    {
        "hypothesis_id": "H4_phi_pattern",
        "head_name": "L10H7",
        "metric": "phi_pattern",
        "predicted_direction": ">",
        "primary": True,
        "architectural": False,
    },
    {
        "hypothesis_id": "H4_conditional_effect",
        "head_name": "L10H7",
        "metric": "conditional_effect",
        "predicted_direction": "<",
        "primary": True,
        "architectural": False,
    },
    {
        "hypothesis_id": "S1_L10H10_interaction_PC",
        "head_name": "L10H10",
        "metric": "interaction_PC",
        "predicted_direction": "<",
        "primary": False,
        "architectural": False,
    },
    {
        "hypothesis_id": "S2_L10H2_phi_context",
        "head_name": "L10H2",
        "metric": "phi_context",
        "predicted_direction": "<",
        "primary": False,
        "architectural": False,
    },
    {
        "hypothesis_id": "secondary_L10H1_phi_context",
        "head_name": "L10H1",
        "metric": "phi_context",
        "predicted_direction": ">",
        "primary": False,
        "architectural": False,
    },
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def boot_seed(*parts) -> int:
    return stable_sha256_seed("phase4b", *parts)


def load_all(full_root: Path) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        p = full_root / f"seed_{seed}" / "factorial_raw.csv"
        df = pd.read_csv(p)
        assert len(df) == N_PER_SEED * 12, f"seed {seed}: rows={len(df)}"
        assert df["head_name"].nunique() == 12
        meta = df.drop_duplicates("example_idx")
        assert len(meta) == N_PER_SEED
        assert meta["template_stratum"].nunique() == N_STRATA
        assert meta["low_level_template_id"].nunique() == 15
        assert set(meta["pattern_type"]) == {"BABA", "ABBA"}
        counts = meta.groupby("template_stratum").size()
        assert (counts == N_PER_STRATUM).all(), counts.value_counts().to_dict()
        # identity checks
        assert float(df["err_H000_vs_intact"].max()) <= 1e-3
        assert float(df["err_H111_vs_conditional"].max()) <= 1e-3
        assert float(df["shapley_residual"].abs().max()) <= 1e-5
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    assert len(out) == N_PER_SEED * 12 * 4
    return out


def head_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["seed", "head_name", "head_role"], sort=False)[METRICS]
        .mean()
        .reset_index()
    )
    return g


def head_aggregate_summary(hs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (head, role), sub in hs.groupby(["head_name", "head_role"], sort=False):
        row = {"head_name": head, "head_role": role, "n_seeds": len(sub)}
        for m in METRICS:
            vals = sub[m].to_numpy(dtype=np.float64)
            row[f"mean_{m}"] = float(np.mean(vals))
            row[f"std_{m}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"n_seeds_pos_{m}"] = int(np.sum(vals > 0))
            row[f"n_seeds_neg_{m}"] = int(np.sum(vals < 0))
            row[f"n_seeds_correct_dir_placeholder_{m}"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def stratum_summary(df: pd.DataFrame, expected_n_strata: int = N_STRATA) -> pd.DataFrame:
    """One row per head × stratum aggregated over seeds."""
    metrics = [
        "conditional_effect",
        "conditional_gain",
        "phi_pattern",
        "phi_value",
        "phi_context",
        "interaction_PC",
        "intact_effect",
    ]
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
        )[metrics]
        .agg(["mean", "count"])
        .reset_index()
    )
    flat = pd.DataFrame(
        {
            "head_name": g["head_name"],
            "head_role": g["head_role"],
            "low_level_template_id": g["low_level_template_id"],
            "pattern_type": g["pattern_type"],
            "template_stratum": g["template_stratum"],
        }
    )
    for m in metrics:
        flat[m] = g[(m, "mean")]
        flat[f"n_{m}"] = g[(m, "count")]
    assert flat.groupby("head_name").size().min() == expected_n_strata
    # n = (#seeds) × (#examples per stratum per seed)
    n_expected = int(flat["n_conditional_effect"].iloc[0])
    assert (flat["n_conditional_effect"] == n_expected).all()
    return flat


def _vectorized_stratified_bootstrap_ci(
    values: np.ndarray,
    strata: np.ndarray,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Fast stratified bootstrap CI (preserves stratum counts)."""
    values = np.asarray(values, dtype=np.float64)
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(strata, return_counts=True)
    # Stack draws: for each stratum, (n_boot, count) indices into that stratum's values
    means = np.zeros(n_bootstrap, dtype=np.float64)
    total_n = 0
    for u, c in zip(unique, counts):
        idxs = np.where(strata == u)[0]
        c = int(c)
        total_n += c
        draws = rng.choice(idxs, size=(n_bootstrap, c), replace=True)
        means += values[draws].sum(axis=1)
    means /= total_n
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return float(np.mean(values)), low, high


def within_stratum_bootstrap_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Within-stratum bootstrap CI for each head × stratum (n=120 across seeds)."""
    rows = []
    metrics = [
        "conditional_effect",
        "conditional_gain",
        "phi_pattern",
        "phi_value",
        "phi_context",
        "interaction_PC",
    ]
    for (hname, stratum), sub in df.groupby(["head_name", "template_stratum"], sort=True):
        role = sub["head_role"].iloc[0]
        tid = int(sub["low_level_template_id"].iloc[0])
        ptype = str(sub["pattern_type"].iloc[0])
        strata = sub["seed"].to_numpy()
        for metric in metrics:
            vals = sub[metric].to_numpy(dtype=np.float64)
            mean, lo, hi = _vectorized_stratified_bootstrap_ci(
                vals,
                strata,
                n_bootstrap=N_BOOT,
                seed=boot_seed("within_stratum", hname, stratum, metric),
            )
            rows.append(
                {
                    "head_name": hname,
                    "head_role": role,
                    "low_level_template_id": tid,
                    "pattern_type": ptype,
                    "template_stratum": stratum,
                    "metric": metric,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n": len(vals),
                    "n_bootstrap": N_BOOT,
                }
            )
    return pd.DataFrame(rows)


def within30_bootstrap_per_seed(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for hname, sub in df[df["seed"] == seed].groupby("head_name"):
            role = sub["head_role"].iloc[0]
            strata = sub["template_stratum"].to_numpy()
            for metric in PRIMARY_BOOT_METRICS:
                vals = sub[metric].to_numpy(dtype=np.float64)
                mean, lo, hi = _vectorized_stratified_bootstrap_ci(
                    vals,
                    strata,
                    n_bootstrap=N_BOOT,
                    seed=boot_seed("within30", seed, hname, metric),
                )
                rows.append(
                    {
                        "seed": seed,
                        "head_name": hname,
                        "head_role": role,
                        "metric": metric,
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "ci_excludes_zero_pos": lo > 0,
                        "ci_excludes_zero_neg": hi < 0,
                        "n": len(vals),
                        "procedure": "within_30_stratum",
                        "n_bootstrap": N_BOOT,
                    }
                )
    return pd.DataFrame(rows)


def template_cluster_bootstrap_per_seed(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for hname, sub in df[df["seed"] == seed].groupby("head_name"):
            role = sub["head_role"].iloc[0]
            tid = sub["low_level_template_id"].to_numpy()
            ptype = sub["pattern_type"].to_numpy()
            for metric in PRIMARY_BOOT_METRICS:
                vals = sub[metric].to_numpy(dtype=np.float64)
                mean, lo, hi = template_cluster_bootstrap_ci(
                    vals,
                    tid,
                    ptype,
                    n_bootstrap=N_BOOT,
                    seed=boot_seed("cluster", seed, hname, metric),
                )
                rows.append(
                    {
                        "seed": seed,
                        "head_name": hname,
                        "head_role": role,
                        "metric": metric,
                        "mean": mean,
                        "ci_low": lo,
                        "ci_high": hi,
                        "ci_excludes_zero_pos": lo > 0,
                        "ci_excludes_zero_neg": hi < 0,
                        "n": len(vals),
                        "procedure": "template_cluster",
                        "n_bootstrap": N_BOOT,
                    }
                )
    return pd.DataFrame(rows)


def hierarchical_cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hname, sub in df.groupby("head_name", sort=False):
        role = sub["head_role"].iloc[0]
        for metric in PRIMARY_BOOT_METRICS:
            mean, lo, hi = hierarchical_template_cluster_bootstrap_ci(
                sub[metric].to_numpy(dtype=np.float64),
                sub["seed"].to_numpy(),
                sub["low_level_template_id"].to_numpy(),
                sub["pattern_type"].to_numpy(),
                n_bootstrap=N_BOOT,
                seed=boot_seed("hier", hname, metric),
            )
            # per-seed means for direction agreement
            seed_means = (
                sub.groupby("seed")[metric].mean().reindex(SEEDS).to_numpy(dtype=np.float64)
            )
            rows.append(
                {
                    "head_name": hname,
                    "head_role": role,
                    "metric": metric,
                    "aggregate_mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "ci_excludes_zero_pos": lo > 0,
                    "ci_excludes_zero_neg": hi < 0,
                    "seed_mean_42": float(seed_means[0]),
                    "seed_mean_1": float(seed_means[1]),
                    "seed_mean_8": float(seed_means[2]),
                    "seed_mean_22": float(seed_means[3]),
                    "n_seeds_pos": int(np.sum(seed_means > 0)),
                    "n_seeds_neg": int(np.sum(seed_means < 0)),
                    "procedure": "hierarchical_template_cluster",
                    "n_bootstrap": N_BOOT,
                    "primary_ci_for_generalization": True,
                }
            )
    return pd.DataFrame(rows)


def baba_abba_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "conditional_effect",
        "conditional_gain",
        "phi_pattern",
        "phi_value",
        "phi_context",
        "interaction_PC",
    ]
    rows = []
    # Aggregate over seeds for each head × pattern
    for hname, hsub in df.groupby("head_name"):
        role = hsub["head_role"].iloc[0]
        for ptype in ("BABA", "ABBA"):
            sub = hsub[hsub["pattern_type"] == ptype]
            row = {
                "head_name": hname,
                "head_role": role,
                "pattern_type": ptype,
                "n": len(sub),
            }
            for m in metrics:
                row[m] = float(sub[m].mean())
            rows.append(row)
        # Paired low-level template differences (BABA − ABBA), averaged over seeds
        for tid in range(15):
            baba = hsub[
                (hsub["low_level_template_id"] == tid) & (hsub["pattern_type"] == "BABA")
            ]
            abba = hsub[
                (hsub["low_level_template_id"] == tid) & (hsub["pattern_type"] == "ABBA")
            ]
            for m in metrics:
                vb = float(baba[m].mean())
                va = float(abba[m].mean())
                rows.append(
                    {
                        "head_name": hname,
                        "head_role": role,
                        "pattern_type": "PAIR_DIFF_BABA_minus_ABBA",
                        "low_level_template_id": tid,
                        "metric": m,
                        "baba_mean": vb,
                        "abba_mean": va,
                        "diff": vb - va,
                        "sign_reversal": (vb > 0 and va < 0) or (vb < 0 and va > 0),
                        "large_asymmetry": abs(vb - va) > max(0.1, 0.5 * max(abs(vb), abs(va), 1e-8)),
                        "n": len(baba) + len(abba),
                    }
                )
    return pd.DataFrame(rows)


def filter_nonoverlap(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["template_stratum"] != EXACT_OVERLAP_STRATUM].copy()


def direction_correct(mean: float, predicted: str) -> bool:
    if predicted == ">":
        return mean > 0
    if predicted == "<":
        return mean < 0
    if predicted == "approx_zero":
        return abs(mean) <= ARCH_TOL
    raise ValueError(predicted)


def ci_supports(lo: float, hi: float, predicted: str) -> bool:
    if predicted == ">":
        return lo > 0
    if predicted == "<":
        return hi < 0
    if predicted == "approx_zero":
        return abs(lo) <= ARCH_TOL and abs(hi) <= ARCH_TOL or (lo <= 0 <= hi and max(abs(lo), abs(hi)) <= 10 * ARCH_TOL)
    raise ValueError(predicted)


def n_strata_correct(stratum_df: pd.DataFrame, head: str, metric: str, predicted: str) -> Tuple[int, int]:
    sub = stratum_df[stratum_df["head_name"] == head]
    n_total = len(sub)
    n_ok = int(sum(direction_correct(float(r[metric]), predicted) for _, r in sub.iterrows()))
    return n_ok, n_total


def evaluate_status(
    *,
    architectural: bool,
    predicted: str,
    mean: float,
    lo: float,
    hi: float,
    n_seeds_ok: int,
    n_strata_ok: int,
    n_strata_total: int,
) -> str:
    if architectural:
        # Numerical tolerance rather than sign criteria
        if abs(mean) <= ARCH_TOL and abs(lo) <= max(ARCH_TOL, abs(mean) + 1e-6) and abs(hi) <= max(ARCH_TOL * 50, 1e-3):
            # Prefer: mean near zero; CI not large in magnitude
            if abs(mean) <= ARCH_TOL and max(abs(lo), abs(hi)) <= 1e-3:
                return "ARCHITECTURAL_INVARIANT_PASS"
        if abs(mean) <= ARCH_TOL and max(abs(lo), abs(hi)) <= 1e-3:
            return "ARCHITECTURAL_INVARIANT_PASS"
        return "ARCHITECTURAL_INVARIANT_FAIL"

    ci_ok = ci_supports(lo, hi, predicted)
    seeds_ok = n_seeds_ok >= 3
    strata_ok = n_strata_ok >= 20
    strong_seeds = n_seeds_ok == 4
    strong_strata = n_strata_ok >= 24

    if ci_ok and seeds_ok and strata_ok:
        if strong_seeds and strong_strata:
            return "STRONGLY_BROADLY_SUPPORTED"
        return "BROADLY_SUPPORTED"
    return "NOT_BROADLY_SUPPORTED"


def build_hypothesis_evaluation(
    df_all: pd.DataFrame,
    hier: pd.DataFrame,
    stratum_all: pd.DataFrame,
    stratum_29: pd.DataFrame,
    hier_29: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for hyp in HYPOTHESES:
        head = hyp["head_name"]
        metric = hyp["metric"]
        pred = hyp["predicted_direction"]

        def pack(hier_df, stratum_df, label_prefix: str):
            hrow = hier_df[(hier_df["head_name"] == head) & (hier_df["metric"] == metric)]
            assert len(hrow) == 1, (head, metric, label_prefix, len(hrow))
            hr = hrow.iloc[0]
            mean, lo, hi = float(hr["aggregate_mean"]), float(hr["ci_low"]), float(hr["ci_high"])
            seed_means = np.array(
                [hr["seed_mean_42"], hr["seed_mean_1"], hr["seed_mean_8"], hr["seed_mean_22"]],
                dtype=np.float64,
            )
            n_seeds_ok = int(sum(direction_correct(float(v), pred) for v in seed_means))
            n_strata_ok, n_strata_total = n_strata_correct(stratum_df, head, metric, pred)
            status = evaluate_status(
                architectural=bool(hyp["architectural"]),
                predicted=pred,
                mean=mean,
                lo=lo,
                hi=hi,
                n_seeds_ok=n_seeds_ok,
                n_strata_ok=n_strata_ok,
                n_strata_total=n_strata_total,
            )
            return {
                "mean": mean,
                "lo": lo,
                "hi": hi,
                "n_seeds_ok": n_seeds_ok,
                "n_strata_ok": n_strata_ok,
                "n_strata_total": n_strata_total,
                "status": status,
            }

        a30 = pack(hier, stratum_all, "all30")
        a29 = pack(hier_29, stratum_29, "nonoverlap29")
        rows.append(
            {
                "hypothesis_id": hyp["hypothesis_id"],
                "head_name": head,
                "metric": metric,
                "predicted_direction": pred,
                "primary": hyp["primary"],
                "aggregate_mean": a30["mean"],
                "hierarchical_cluster_ci_low": a30["lo"],
                "hierarchical_cluster_ci_high": a30["hi"],
                "n_seeds_correct_direction": a30["n_seeds_ok"],
                "n_strata_correct_direction": a30["n_strata_ok"],
                "n_strata_total": a30["n_strata_total"],
                "all30_status": a30["status"],
                "nonoverlap29_status": a29["status"],
                "nonoverlap29_mean": a29["mean"],
                "nonoverlap29_ci_low": a29["lo"],
                "nonoverlap29_ci_high": a29["hi"],
            }
        )
    return pd.DataFrame(rows)


def scientific_verdict(hyp_eval: pd.DataFrame) -> str:
    """Predefined rule — do not modify after results."""
    # Primary directional quantities (exclude architectural invariants for count)
    primary = hyp_eval[hyp_eval["primary"] == True]  # noqa: E712
    # Group into H1–H4 hypothesis families
    families = {
        "H1": primary[primary["hypothesis_id"].str.startswith("H1_")],
        "H2": primary[primary["hypothesis_id"].str.startswith("H2_")],
        "H3": primary[primary["hypothesis_id"].str.startswith("H3_")],
        "H4": primary[primary["hypothesis_id"].str.startswith("H4_")],
    }

    def family_supported(sub: pd.DataFrame) -> Tuple[bool, bool]:
        """Returns (broadly_supported, strongly_supported) for a family."""
        statuses = sub["all30_status"].tolist()
        # Architectural rows: PASS counts as support for that component
        ok = []
        strong = []
        for st in statuses:
            if st in ("BROADLY_SUPPORTED", "STRONGLY_BROADLY_SUPPORTED", "ARCHITECTURAL_INVARIANT_PASS"):
                ok.append(True)
            else:
                ok.append(False)
            if st in ("STRONGLY_BROADLY_SUPPORTED", "ARCHITECTURAL_INVARIANT_PASS"):
                strong.append(True)
            else:
                strong.append(False)
        # Family broadly supported if ALL component quantities supported
        broadly = all(ok) and len(ok) > 0
        # Strongly if all components at least broadly and all strong-tier
        strongly = broadly and all(strong)
        return broadly, strongly

    broadly_flags = []
    strongly_flags = []
    for name, sub in families.items():
        b, s = family_supported(sub)
        broadly_flags.append(b)
        strongly_flags.append(s)

    n_broad = sum(broadly_flags)
    n_strong = sum(strongly_flags)

    if n_broad == 4 and n_strong >= 3:
        return "STRONG BROAD-TEMPLATE REPLICATION"
    if n_broad >= 2:
        return "BROAD-TEMPLATE PARTIAL"
    return "BROAD-TEMPLATE FAILED"


def write_head_report(
    path: Path,
    head: str,
    title: str,
    hyp_eval: pd.DataFrame,
    hier: pd.DataFrame,
    stratum: pd.DataFrame,
    baba_abba: pd.DataFrame,
    framing: str = "primary",
) -> None:
    lines = [f"# {title}", "", f"Framing: **{framing}**", ""]
    he = hyp_eval[hyp_eval["head_name"] == head]
    lines.append("## Hypothesis quantities")
    lines.append("")
    if len(he):
        lines.append(he.to_string(index=False))
    else:
        lines.append("(no frozen hypothesis rows)")
    lines.append("")
    h = hier[hier["head_name"] == head]
    lines.append("## Hierarchical template-cluster aggregate (all 30 strata)")
    lines.append("")
    show = h[
        h["metric"].isin(
            [
                "conditional_effect",
                "conditional_gain",
                "phi_pattern",
                "phi_value",
                "phi_context",
                "interaction_PC",
            ]
        )
    ]
    lines.append(show.to_string(index=False))
    lines.append("")
    st = stratum[stratum["head_name"] == head]
    lines.append("## Stratum-level means (n=120 per stratum)")
    lines.append("")
    lines.append(
        st[
            [
                "template_stratum",
                "conditional_effect",
                "conditional_gain",
                "phi_pattern",
                "phi_value",
                "phi_context",
                "interaction_PC",
            ]
        ].to_string(index=False)
    )
    lines.append("")
    ba = baba_abba[
        (baba_abba["head_name"] == head) & (baba_abba["pattern_type"].isin(["BABA", "ABBA"]))
    ]
    lines.append("## BABA vs ABBA aggregates")
    lines.append("")
    lines.append(ba.to_string(index=False))
    lines.append("")
    lines.append(
        "_Do not interpret N=60 smoke; this report uses full N=900×4. "
        "Secondary hypotheses are confirmatory only._"
    )
    path.write_text("\n".join(lines) + "\n")


def make_plots(out: Path, hier: pd.DataFrame, stratum: pd.DataFrame, baba_abba: pd.DataFrame) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Head Shapley decomposition
    metrics = ["phi_pattern", "phi_value", "phi_context"]
    heads = sorted(hier["head_name"].unique())
    # preserve a sensible order
    prefer = [
        "L9H0",
        "L9H7",
        "L10H1",
        "L10H2",
        "L10H6",
        "L10H10",
        "L11H2",
        "L11H9",
        "L10H7",
        "L11H10",
        "L10H3",
        "L10H5",
    ]
    heads = [h for h in prefer if h in heads] + [h for h in heads if h not in prefer]
    mat = np.array(
        [
            [
                float(
                    hier[(hier["head_name"] == h) & (hier["metric"] == m)]["aggregate_mean"].iloc[0]
                )
                for m in metrics
            ]
            for h in heads
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(heads))
    w = 0.25
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1) * w, mat[:, i], width=w, label=m)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(heads, rotation=45, ha="right")
    ax.set_ylabel("Equal-seed mean")
    ax.set_title("Shapley decomposition (hierarchical point estimates)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "head_shapley_decomposition.png", dpi=150)
    plt.close(fig)

    # 2. L10H2 vs L10H10
    fig, ax = plt.subplots(figsize=(8, 4))
    labels, means, yerr_lo, yerr_hi, colors = [], [], [], [], []
    color_map = {"L10H2": "C0", "L10H10": "C1"}
    for head in ("L10H2", "L10H10"):
        sub = hier[hier["head_name"] == head]
        for m in ("phi_pattern", "phi_context", "conditional_gain"):
            r = sub[sub["metric"] == m].iloc[0]
            labels.append(f"{head}\n{m}")
            means.append(float(r["aggregate_mean"]))
            yerr_lo.append(float(r["aggregate_mean"] - r["ci_low"]))
            yerr_hi.append(float(r["ci_high"] - r["aggregate_mean"]))
            colors.append(color_map[head])
    x = np.arange(len(labels))
    ax.bar(x, means, color=colors, yerr=np.vstack([yerr_lo, yerr_hi]), capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("L10H2 vs L10H10 (hierarchical cluster CI)")
    fig.tight_layout()
    fig.savefig(fig_dir / "L10H2_vs_L10H10.png", dpi=150)
    plt.close(fig)

    def stratum_bar(head: str, metric: str, fname: str, title: str):
        sub = stratum[stratum["head_name"] == head].sort_values("template_stratum")
        fig, ax = plt.subplots(figsize=(12, 4))
        colors = ["C0" if p == "BABA" else "C1" for p in sub["pattern_type"]]
        ax.bar(range(len(sub)), sub[metric], color=colors)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["template_stratum"], rotation=90, fontsize=7)
        ax.set_title(title)
        ax.set_ylabel(metric)
        fig.tight_layout()
        fig.savefig(fig_dir / fname, dpi=150)
        plt.close(fig)

    stratum_bar("L9H7", "phi_context", "L9H7_phi_context_by_stratum.png", "L9H7 φ_context by stratum")
    stratum_bar("L10H2", "phi_pattern", "L10H2_phi_pattern_by_stratum.png", "L10H2 φ_pattern by stratum")
    # L10H10 pattern/context
    sub = stratum[stratum["head_name"] == "L10H10"].sort_values("template_stratum")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, metric in zip(axes, ["phi_pattern", "phi_context"]):
        colors = ["C0" if p == "BABA" else "C1" for p in sub["pattern_type"]]
        ax.bar(range(len(sub)), sub[metric], color=colors)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(metric)
    axes[1].set_xticks(range(len(sub)))
    axes[1].set_xticklabels(sub["template_stratum"], rotation=90, fontsize=7)
    axes[0].set_title("L10H10 φ_pattern / φ_context by stratum")
    fig.tight_layout()
    fig.savefig(fig_dir / "L10H10_phi_pattern_context_by_stratum.png", dpi=150)
    plt.close(fig)

    # BABA vs ABBA comparison for key heads
    fig, ax = plt.subplots(figsize=(8, 4))
    key_heads = ["L9H7", "L10H2", "L10H10", "L10H7"]
    xlabels = []
    vals_b, vals_a = [], []
    for h in key_heads:
        for m in ["phi_pattern", "phi_context"]:
            b = baba_abba[
                (baba_abba["head_name"] == h)
                & (baba_abba["pattern_type"] == "BABA")
            ]
            a = baba_abba[
                (baba_abba["head_name"] == h)
                & (baba_abba["pattern_type"] == "ABBA")
            ]
            if len(b) and m in b.columns:
                xlabels.append(f"{h}\n{m}")
                vals_b.append(float(b[m].iloc[0]))
                vals_a.append(float(a[m].iloc[0]))
    x = np.arange(len(xlabels))
    ax.bar(x - 0.2, vals_b, width=0.4, label="BABA")
    ax.bar(x + 0.2, vals_a, width=0.4, label="ABBA")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.legend()
    ax.set_title("BABA vs ABBA (key heads)")
    fig.tight_layout()
    fig.savefig(fig_dir / "baba_vs_abba_comparison.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--full-root",
        default="results/revision_phase4/full",
    )
    p.add_argument("--n-bootstrap", type=int, default=None)
    args = p.parse_args()

    if args.n_bootstrap is not None:
        globals()["N_BOOT"] = int(args.n_bootstrap)

    full_root = Path(args.full_root)
    if not full_root.is_absolute():
        full_root = REPO / full_root
    out = full_root / "aggregate"
    out.mkdir(parents=True, exist_ok=True)

    print("Loading full factorial raw...", flush=True)
    df = load_all(full_root)
    df29 = filter_nonoverlap(df)

    print("Head × seed summary...", flush=True)
    hs = head_seed_summary(df)
    hs.to_csv(out / "head_seed_summary.csv", index=False)

    print("Head aggregate summary...", flush=True)
    ha = head_aggregate_summary(hs)
    ha.to_csv(out / "head_aggregate_summary.csv", index=False)

    print("Stratum summary (n=120)...", flush=True)
    st = stratum_summary(df, expected_n_strata=30)
    st.to_csv(out / "stratum_summary.csv", index=False)
    st29 = stratum_summary(df29, expected_n_strata=29)
    st29.to_csv(out / "stratum_summary_nonoverlap29.csv", index=False)

    print("Within-stratum bootstrap (head × 30 strata)...")
    wsb = within_stratum_bootstrap_summary(df)
    wsb.to_csv(out / "stratum_bootstrap_summary.csv", index=False)
    wsb.to_csv(out / "within_stratum_bootstrap_summary.csv", index=False)

    print("Within-30-stratum bootstrap per seed...")
    w30 = within30_bootstrap_per_seed(df)
    # also save combined name requested
    w30.to_csv(out / "within_stratum_bootstrap_per_seed.csv", index=False)

    print("Template-cluster bootstrap per seed...")
    tc = template_cluster_bootstrap_per_seed(df)
    tc.to_csv(out / "template_cluster_bootstrap_summary.csv", index=False)

    print("Hierarchical template-cluster (PRIMARY CI)...")
    hier = hierarchical_cluster_summary(df)
    hier.to_csv(out / "hierarchical_template_cluster_summary.csv", index=False)

    print("Hierarchical on 29 non-overlap strata...")
    # Point estimates use all 29 strata (exclude exact-overlap T00_BABA only).
    # Hierarchical CI uses the 14 fully paired templates T01–T14 (28 strata),
    # since T00 lacks BABA after exclusion and cluster pairing requires both patterns.
    df_paired29 = df[df["low_level_template_id"] != 0].copy()
    assert df_paired29["template_stratum"].nunique() == 28
    hier_29_rows = []
    for hname, sub in df29.groupby("head_name", sort=False):
        role = sub["head_role"].iloc[0]
        for metric in PRIMARY_BOOT_METRICS:
            seed_means = (
                sub.groupby("seed")[metric].mean().reindex(SEEDS).to_numpy(dtype=np.float64)
            )
            point = float(np.mean(seed_means))
            sub28 = df_paired29[df_paired29["head_name"] == hname]
            tid = sub28["low_level_template_id"].to_numpy().astype(np.int64) - 1
            _, lo, hi = hierarchical_template_cluster_bootstrap_ci(
                sub28[metric].to_numpy(dtype=np.float64),
                sub28["seed"].to_numpy(),
                tid,
                sub28["pattern_type"].to_numpy(),
                n_bootstrap=N_BOOT,
                seed=boot_seed("hier29", hname, metric),
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
                    "n_bootstrap": N_BOOT,
                }
            )
    hier_29 = pd.DataFrame(hier_29_rows)
    hier_29.to_csv(out / "hierarchical_template_cluster_nonoverlap29.csv", index=False)

    # Nonoverlap summary table (equal-seed means on 29)
    nonoverlap_rows = []
    for hname, sub in df29.groupby("head_name"):
        role = sub["head_role"].iloc[0]
        row = {"head_name": hname, "head_role": role, "n_strata": 29}
        for m in PRIMARY_BOOT_METRICS:
            seed_means = sub.groupby("seed")[m].mean().reindex(SEEDS)
            row[f"mean_{m}"] = float(seed_means.mean())
        nonoverlap_rows.append(row)
    pd.DataFrame(nonoverlap_rows).to_csv(out / "nonoverlap_29strata_summary.csv", index=False)

    print("BABA/ABBA...")
    ba = baba_abba_summary(df)
    ba.to_csv(out / "baba_abba_summary.csv", index=False)

    print("Factorial interactions / endpoint marginals...")
    inter_cols = [
        "main_P",
        "main_V",
        "main_C",
        "interaction_PV",
        "interaction_PC",
        "interaction_VC",
        "interaction_PVC",
    ]
    marg_cols = [
        "pattern_effect_B_VB",
        "value_effect_B_PB",
        "context_effect_PB_VB",
        "pattern_effect_A_VA",
        "value_effect_A_PA",
        "context_effect_PA_VA",
    ]
    inter = (
        df.groupby(["head_name", "head_role"], sort=False)[inter_cols].mean().reset_index()
    )
    inter.to_csv(out / "factorial_interactions.csv", index=False)
    marg = (
        df.groupby(["head_name", "head_role"], sort=False)[marg_cols].mean().reset_index()
    )
    marg.to_csv(out / "endpoint_marginal_effects.csv", index=False)

    print("Value contribution...")
    val_rows = []
    for hname, sub in hier.groupby("head_name"):
        def get(m):
            return sub[sub["metric"] == m].iloc[0]

        pv, pp, pc = get("phi_value"), get("phi_pattern"), get("phi_context")
        val_rows.append(
            {
                "head_name": hname,
                "abs_phi_value": abs(pv["aggregate_mean"]),
                "abs_phi_pattern": abs(pp["aggregate_mean"]),
                "abs_phi_context": abs(pc["aggregate_mean"]),
                "phi_value_ci_low": pv["ci_low"],
                "phi_value_ci_high": pv["ci_high"],
                "phi_value_much_smaller": abs(pv["aggregate_mean"])
                < 0.1 * max(abs(pp["aggregate_mean"]), abs(pc["aggregate_mean"]), 1e-8),
            }
        )
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(out / "value_contribution_summary.csv", index=False)

    print("Hypothesis evaluation...")
    hyp_eval = build_hypothesis_evaluation(df, hier, st, st29, hier_29)
    hyp_eval.to_csv(out / "hypothesis_evaluation.csv", index=False)

    verdict = scientific_verdict(hyp_eval)
    (out / "scientific_verdict.md").write_text(
        f"# Scientific verdict\n\n**{verdict}**\n\n"
        "Rule (pre-registered):\n"
        "- STRONG BROAD-TEMPLATE REPLICATION: all 4 primary families broadly supported "
        "and ≥3 strongly supported.\n"
        "- BROAD-TEMPLATE PARTIAL: ≥2 primary families broadly supported.\n"
        "- BROAD-TEMPLATE FAILED: <2 primary families broadly supported.\n"
        "Secondary hypotheses do not determine the main verdict.\n"
    )

    print("Head reports...")
    write_head_report(
        out / "L9H7_broad_template_report.md",
        "L9H7",
        "L9H7 broad-template report",
        hyp_eval,
        hier,
        st,
        ba,
        framing="primary H1 — context recruitment",
    )
    write_head_report(
        out / "L10H2_broad_template_report.md",
        "L10H2",
        "L10H2 broad-template report",
        hyp_eval,
        hier,
        st,
        ba,
        framing="primary H2 — pattern recruitment (+ secondary S2)",
    )
    write_head_report(
        out / "L10H10_broad_template_report.md",
        "L10H10",
        "L10H10 broad-template report",
        hyp_eval,
        hier,
        st,
        ba,
        framing="primary H3 — cancellation (+ secondary S1)",
    )
    write_head_report(
        out / "L10H7_broad_template_report.md",
        "L10H7",
        "L10H7 broad-template report",
        hyp_eval,
        hier,
        st,
        ba,
        framing="primary H4 — adaptive-but-suppressive",
    )
    write_head_report(
        out / "L10H1_broad_template_report.md",
        "L10H1",
        "L10H1 broad-template report",
        hyp_eval,
        hier,
        st,
        ba,
        framing="secondary / suggestive",
    )

    print("Plots...")
    make_plots(out, hier, st, ba)

    # Raw artifact index (omit full CSVs from ZIP later)
    raw_index = []
    for seed in SEEDS:
        p = full_root / f"seed_{seed}" / "factorial_raw.csv"
        raw = pd.read_csv(p, nrows=5)
        raw_index.append(
            {
                "path": str(p.relative_to(REPO)),
                "row_count": N_PER_SEED * 12,
                "schema": list(raw.columns),
                "sha256": file_sha256(p),
            }
        )
    (out / "raw_factorial_index.json").write_text(json.dumps(raw_index, indent=2))
    # representative samples
    df.head(20).to_csv(out / "factorial_raw_representative_sample.csv", index=False)

    # Mechanistic heterogeneity (descriptive, no classifier)
    (out / "mechanistic_heterogeneity_note.md").write_text(
        "# Mechanistic heterogeneity (descriptive)\n\n"
        "Do not force one taxonomy. Compare genuine backup heads descriptively:\n\n"
        "- L9H7: context-led (φ_P≈0, φ_V≈0, φ_C).\n"
        "- L10H2: pattern recruitment with possible opposing context.\n\n"
        "See hierarchical_template_cluster_summary.csv and head reports.\n"
    )

    meta = {
        "experiment": "canonical-template / public-implementation lexical distribution",
        "n_per_seed": N_PER_SEED,
        "seeds": SEEDS,
        "n_bootstrap": N_BOOT,
        "verdict": verdict,
        "exact_overlap_excluded": EXACT_OVERLAP_STRATUM,
        "arch_tol": ARCH_TOL,
    }
    (out / "aggregate_config.json").write_text(json.dumps(meta, indent=2))

    print(f"VERDICT: {verdict}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
