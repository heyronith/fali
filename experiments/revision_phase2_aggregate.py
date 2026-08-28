"""
Phase 2B/3B aggregation: merge Phase-1/2, bootstrap CIs, FALI↔causal,
predictive CV, contrast reports, and exploratory plots.

Reads:
  results/revision_phase2/full/seed_*/phase2_combined_raw.csv (or fali+causal)
  results/revision_phase1/full/seed_*/candidate_raw.csv
  results/revision_phase1/full/seed_*/dataset_metadata.csv

Writes:
  results/revision_phase2/full/aggregate/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from revision_stats import pearson_spearman, stratified_bootstrap_ci  # noqa: E402

SEEDS = [42, 1, 8, 22]
N_BOOT = 10000
ZERO_TOL = 1e-8  # exclude near-zero for sign agreement
RIDGE_ALPHA = 1.0  # fixed; no test-fold tuning

BOOT_METRICS = [
    "delta_attention",
    "delta_target_score",
    "delta_competing_lse",
    "delta_margin",
    "fali_target",
    "fali_competing",
    "causal_target",
    "causal_competing",
    "causal_total",
    "pattern_clean_patch_effect",
    "q_clean_effect",
    "k_clean_effect",
    "qk_clean_effect",
    "conditional_effect",
    "conditional_gain",
]

FEATURE_SETS = {
    "A_delta_attention": ["delta_attention"],
    "B_delta_margin": ["delta_margin"],
    "C_raw_scores": ["delta_target_score", "delta_competing_lse"],
    "D_fali": ["fali_target", "fali_competing"],
}

TARGETS = ["conditional_gain", "conditional_effect"]

REPORTING_PANEL = ["L10H2", "L10H1", "L10H10", "L9H7", "L10H7", "L10H6"]


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


def load_phase2_seed(full_root: Path, seed: int) -> pd.DataFrame:
    d = full_root / f"seed_{seed}"
    combined = d / "phase2_combined_raw.csv"
    if combined.exists():
        df = pd.read_csv(combined)
    else:
        desc = pd.read_csv(d / "fali_raw.csv")
        causal = pd.read_csv(d / "causal_score_hybrid_raw.csv")
        df = desc.merge(
            causal,
            on=["seed", "example_idx", "head_name", "layer", "head", "head_role"],
            how="inner",
            validate="one_to_one",
        )
    if "seed" not in df.columns:
        df["seed"] = seed
    assert len(df) == 200 * 12, f"seed {seed}: expected 2400 rows, got {len(df)}"
    assert df["head_name"].nunique() == 12
    return df


def load_phase1_seed(phase1_root: Path, seed: int) -> pd.DataFrame:
    raw = pd.read_csv(phase1_root / f"seed_{seed}" / "candidate_raw.csv")
    meta = pd.read_csv(phase1_root / f"seed_{seed}" / "dataset_metadata.csv")
    need = ["example_idx", "intact_effect", "conditional_effect", "conditional_gain", "head_name"]
    for c in need:
        assert c in raw.columns, f"Phase-1 missing {c}"
    raw = raw[need].copy()
    raw["seed"] = seed
    meta = meta[["example_idx", "template_id"]].copy()
    raw = raw.merge(meta, on="example_idx", how="left", validate="many_to_one")
    assert raw["template_id"].notna().all()
    return raw[
        [
            "seed",
            "example_idx",
            "head_name",
            "template_id",
            "intact_effect",
            "conditional_effect",
            "conditional_gain",
        ]
    ]


def merge_all(full_root: Path, phase1_root: Path) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        p2 = load_phase2_seed(full_root, seed)
        p1 = load_phase1_seed(phase1_root, seed)
        m = p2.merge(
            p1,
            on=["seed", "example_idx", "head_name"],
            how="inner",
            validate="one_to_one",
        )
        assert len(m) == 2400, f"seed {seed} merge rows={len(m)}"
        frames.append(m)
    out = pd.concat(frames, ignore_index=True)
    assert len(out) == 9600, f"expected 9600, got {len(out)}"
    return out


def make_bootstrap(df: pd.DataFrame, base_seed: int = 42) -> pd.DataFrame:
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
                    seed=boot_seed(base_seed, seed, head, metric),
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
                        "n": len(vals),
                        "n_bootstrap": N_BOOT,
                    }
                )
    return pd.DataFrame(rows)


def head_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "delta_attention",
        "delta_target_score",
        "delta_competing_lse",
        "delta_margin",
        "fali_target",
        "fali_competing",
        "fali_total_delta",
        "causal_target",
        "causal_competing",
        "causal_total",
        "pattern_clean_patch_effect",
        "q_clean_effect",
        "k_clean_effect",
        "qk_clean_effect",
        "intact_effect",
        "conditional_effect",
        "conditional_gain",
    ]
    g = (
        df.groupby(["seed", "head_name", "head_role"], sort=False)[metrics]
        .mean()
        .reset_index()
    )
    return g


def head_aggregate_summary(hs: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in hs.columns if c not in ("seed", "head_name", "head_role")]
    rows = []
    for (head, role), sub in hs.groupby(["head_name", "head_role"], sort=False):
        row = {"head_name": head, "head_role": role, "n_seeds": len(sub)}
        for m in metrics:
            vals = sub[m].to_numpy(dtype=np.float64)
            row[f"mean_{m}"] = float(np.mean(vals))
            row[f"std_{m}"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def sign_agree(x: np.ndarray, y: np.ndarray, tol: float = ZERO_TOL) -> Dict[str, float]:
    mask = (np.abs(x) > tol) & (np.abs(y) > tol)
    if mask.sum() == 0:
        return {"n_compared": 0, "sign_agreement": float("nan")}
    return {
        "n_compared": int(mask.sum()),
        "sign_agreement": float(np.mean(np.sign(x[mask]) == np.sign(y[mask]))),
    }


def fali_vs_causal(df: pd.DataFrame, hs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Across-head per seed (head means)
    for seed in SEEDS:
        sub = hs[hs["seed"] == seed]
        for xa, ya, label in [
            ("fali_target", "causal_target", "target"),
            ("fali_competing", "causal_competing", "competing"),
        ]:
            corr = pearson_spearman(sub[xa], sub[ya])
            sa = sign_agree(sub[xa].to_numpy(), sub[ya].to_numpy())
            rows.append(
                {
                    "level": "across_head",
                    "seed": seed,
                    "head_name": "ALL",
                    "pair": label,
                    **corr,
                    **sa,
                }
            )
    # Across-seed head means
    agg = (
        hs.groupby("head_name", sort=False)[
            ["fali_target", "fali_competing", "causal_target", "causal_competing"]
        ]
        .mean()
        .reset_index()
    )
    for xa, ya, label in [
        ("fali_target", "causal_target", "target"),
        ("fali_competing", "causal_competing", "competing"),
    ]:
        corr = pearson_spearman(agg[xa], agg[ya])
        sa = sign_agree(agg[xa].to_numpy(), agg[ya].to_numpy())
        rows.append(
            {
                "level": "across_head_seedmean",
                "seed": "all",
                "head_name": "ALL",
                "pair": label,
                **corr,
                **sa,
            }
        )

    # Within-head per seed
    within_rows = []
    for seed in SEEDS:
        for head, sub in df[df["seed"] == seed].groupby("head_name"):
            for xa, ya, label in [
                ("fali_target", "causal_target", "target"),
                ("fali_competing", "causal_competing", "competing"),
            ]:
                corr = pearson_spearman(sub[xa], sub[ya])
                sa = sign_agree(sub[xa].to_numpy(), sub[ya].to_numpy())
                within_rows.append(
                    {
                        "level": "within_head",
                        "seed": seed,
                        "head_name": head,
                        "pair": label,
                        **corr,
                        **sa,
                    }
                )
    rows.extend(within_rows)

    # Aggregate within-head correlations across seeds (mean of per-seed r; not pooled)
    wr = pd.DataFrame(within_rows)
    for head in wr["head_name"].unique():
        for label in ["target", "competing"]:
            sub = wr[(wr["head_name"] == head) & (wr["pair"] == label)]
            pr = sub["pearson_r"].dropna().to_numpy(dtype=np.float64)
            sr = sub["spearman_r"].dropna().to_numpy(dtype=np.float64)
            rows.append(
                {
                    "level": "within_head_across_seeds_mean_r",
                    "seed": "all",
                    "head_name": head,
                    "pair": label,
                    "pearson_r": float(np.mean(pr)) if pr.size else None,
                    "pearson_p": None,
                    "spearman_r": float(np.mean(sr)) if sr.size else None,
                    "spearman_p": None,
                    "n_compared": int(sub["n_compared"].sum()),
                    "sign_agreement": float(sub["sign_agreement"].mean())
                    if sub["sign_agreement"].notna().any()
                    else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def fali_vs_function(hs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        sub = hs[hs["seed"] == seed]
        for x in [
            "delta_attention",
            "delta_margin",
            "fali_target",
            "fali_competing",
            "causal_total",
        ]:
            for y in ["conditional_gain", "conditional_effect"]:
                corr = pearson_spearman(sub[x], sub[y])
                rows.append(
                    {
                        "level": "across_head",
                        "seed": seed,
                        "x": x,
                        "y": y,
                        **corr,
                    }
                )
    agg = (
        hs.groupby("head_name", sort=False)[
            [
                "delta_attention",
                "delta_margin",
                "fali_target",
                "fali_competing",
                "causal_total",
                "conditional_gain",
                "conditional_effect",
            ]
        ]
        .mean()
        .reset_index()
    )
    for x in [
        "delta_attention",
        "delta_margin",
        "fali_target",
        "fali_competing",
        "causal_total",
    ]:
        for y in ["conditional_gain", "conditional_effect"]:
            corr = pearson_spearman(agg[x], agg[y])
            rows.append(
                {"level": "across_head_seedmean", "seed": "all", "x": x, "y": y, **corr}
            )
    return pd.DataFrame(rows)


def fit_predict_ridge(X_train, y_train, X_test, y_test, alpha: float = RIDGE_ALPHA):
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)

    # Drop constant features on the training fold only (no leakage).
    std = X_train.std(axis=0)
    keep = std > 1e-12
    if not np.any(keep):
        # Degenerate fold (e.g. all-zero FALI on same-layer head): predict train mean
        pred = np.full_like(y_test, float(np.mean(y_train)))
        return {
            "r2": float(r2_score(y_test, pred)),
            "mae": float(mean_absolute_error(y_test, pred)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "n_features_used": 0,
        }
    X_train = X_train[:, keep]
    X_test = X_test[:, keep]

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features_used": int(keep.sum()),
    }


def predictive_cv(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows = []
    heads = sorted(df["head_name"].unique())
    # A. Leave-one-head-out
    for held in heads:
        train = df[df["head_name"] != held]
        test = df[df["head_name"] == held]
        for target in TARGETS:
            for fname, feats in FEATURE_SETS.items():
                m = fit_predict_ridge(
                    train[feats].to_numpy(),
                    train[target].to_numpy(),
                    test[feats].to_numpy(),
                    test[target].to_numpy(),
                )
                fold_rows.append(
                    {
                        "cv": "leave_one_head_out",
                        "fold": held,
                        "target": target,
                        "feature_set": fname,
                        **m,
                    }
                )
    # B. Leave-one-seed-out
    for held in SEEDS:
        train = df[df["seed"] != held]
        test = df[df["seed"] == held]
        for target in TARGETS:
            for fname, feats in FEATURE_SETS.items():
                m = fit_predict_ridge(
                    train[feats].to_numpy(),
                    train[target].to_numpy(),
                    test[feats].to_numpy(),
                    test[target].to_numpy(),
                )
                fold_rows.append(
                    {
                        "cv": "leave_one_seed_out",
                        "fold": str(held),
                        "target": target,
                        "feature_set": fname,
                        **m,
                    }
                )
    folds = pd.DataFrame(fold_rows)

    # Summary + deltas vs FALI
    sum_rows = []
    for (cv, target, fname), sub in folds.groupby(
        ["cv", "target", "feature_set"], sort=False
    ):
        sum_rows.append(
            {
                "cv": cv,
                "target": target,
                "feature_set": fname,
                "mean_r2": float(sub["r2"].mean()),
                "std_r2": float(sub["r2"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "mean_mae": float(sub["mae"].mean()),
                "std_mae": float(sub["mae"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "n_folds": len(sub),
            }
        )
    summary = pd.DataFrame(sum_rows)

    # Fold-level FALI advantage
    adv_rows = []
    for (cv, target, fold), sub in folds.groupby(["cv", "target", "fold"], sort=False):
        d = sub.set_index("feature_set")
        if "D_fali" not in d.index:
            continue
        for base in ["A_delta_attention", "B_delta_margin", "C_raw_scores"]:
            if base not in d.index:
                continue
            adv_rows.append(
                {
                    "cv": cv,
                    "target": target,
                    "fold": fold,
                    "baseline": base,
                    "delta_r2_fali_minus_base": float(d.loc["D_fali", "r2"] - d.loc[base, "r2"]),
                    "delta_mae_fali_minus_base": float(
                        d.loc["D_fali", "mae"] - d.loc[base, "mae"]
                    ),
                    "fali_r2": float(d.loc["D_fali", "r2"]),
                    "base_r2": float(d.loc[base, "r2"]),
                    "fali_mae": float(d.loc["D_fali", "mae"]),
                    "base_mae": float(d.loc[base, "mae"]),
                }
            )
    advantage = pd.DataFrame(adv_rows)
    return folds, summary, advantage


def causal_baseline_comparison(hs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        sub = hs[hs["seed"] == seed]
        for head, r in sub.groupby("head_name"):
            row = r.iloc[0]
            pairs = [
                ("causal_total", "pattern_clean_patch_effect"),
                ("causal_total", "qk_clean_effect"),
                ("pattern_clean_patch_effect", "qk_clean_effect"),
                ("q_clean_effect", "k_clean_effect"),
                ("q_clean_effect", "qk_clean_effect"),
                ("k_clean_effect", "qk_clean_effect"),
            ]
            for a, b in pairs:
                rows.append(
                    {
                        "seed": seed,
                        "head_name": head,
                        "metric_a": a,
                        "metric_b": b,
                        "mean_a": float(row[a]),
                        "mean_b": float(row[b]),
                        "diff_a_minus_b": float(row[a] - row[b]),
                        "abs_diff": float(abs(row[a] - row[b])),
                    }
                )
    pair_df = pd.DataFrame(rows)
    # Across-head correlations per seed for key pairs
    corr_rows = []
    for seed in SEEDS:
        sub = hs[hs["seed"] == seed]
        for a, b in [
            ("causal_total", "pattern_clean_patch_effect"),
            ("causal_total", "qk_clean_effect"),
            ("pattern_clean_patch_effect", "qk_clean_effect"),
        ]:
            corr = pearson_spearman(sub[a], sub[b])
            corr_rows.append({"seed": seed, "metric_a": a, "metric_b": b, **corr})
    return pair_df, pd.DataFrame(corr_rows)


def template_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "delta_attention",
        "fali_target",
        "fali_competing",
        "causal_target",
        "causal_competing",
        "causal_total",
        "conditional_effect",
        "conditional_gain",
    ]
    g = (
        df.groupby(["seed", "template_id", "head_name", "head_role"], sort=False)[metrics]
        .mean()
        .reset_index()
    )
    return g


def write_contrast_reports(df: pd.DataFrame, hs: pd.DataFrame, out: Path) -> Dict:
    agg = head_aggregate_summary(hs)

    def head_means(name: str) -> pd.Series:
        return agg[agg["head_name"] == name].iloc[0]

    h2 = head_means("L10H2")
    h10 = head_means("L10H10")
    lines = [
        "# L10H2 vs L10H10",
        "",
        "Means across 4 seeds (head-level). N=200/seed. Not post-hoc thresholding.",
        "",
        "| metric | L10H2 | L10H10 |",
        "|--------|-------|--------|",
    ]
    for m in [
        "conditional_effect",
        "conditional_gain",
        "delta_attention",
        "delta_target_score",
        "delta_competing_lse",
        "delta_margin",
        "fali_target",
        "fali_competing",
        "causal_target",
        "causal_competing",
        "causal_total",
        "pattern_clean_patch_effect",
        "q_clean_effect",
        "k_clean_effect",
        "qk_clean_effect",
    ]:
        lines.append(
            f"| {m} | {h2[f'mean_{m}']:.4f} | {h10[f'mean_{m}']:.4f} |"
        )
    # Distinction check
    attn_close = abs(h2["mean_delta_attention"] - h10["mean_delta_attention"]) < 0.05
    fali_sep = abs(h2["mean_fali_target"] - h10["mean_fali_target"])
    gain_sep = abs(h2["mean_conditional_gain"] - h10["mean_conditional_gain"])
    lines += [
        "",
        "## Interpretation (predefined question)",
        f"- |Δattn L10H2−L10H10| = {abs(h2['mean_delta_attention'] - h10['mean_delta_attention']):.4f}",
        f"- |fali_target L10H2−L10H10| = {fali_sep:.4f}",
        f"- |conditional_gain L10H2−L10H10| = {gain_sep:.4f}",
        f"- Attention spikes similarly large? {'YES' if attn_close else 'NO (also differ in Δattn)'}",
        f"- Functional gain differs (L10H2 recruit vs L10H10 downregulated): "
        f"L10H2 gain={h2['mean_conditional_gain']:.4f}, L10H10 gain={h10['mean_conditional_gain']:.4f}",
    ]
    (out / "L10H2_vs_L10H10.md").write_text("\n".join(lines) + "\n")

    # L9H7 scope
    l9 = df[df["head_name"] == "L9H7"]
    assert len(l9) == 800
    scope = {
        "n": len(l9),
        "max_abs_delta_attention": float(l9["delta_attention"].abs().max()),
        "max_abs_delta_target_score": float(l9["delta_target_score"].abs().max()),
        "max_abs_fali_target": float(l9["fali_target"].abs().max()),
        "max_abs_fali_competing": float(l9["fali_competing"].abs().max()),
        "max_abs_causal_total": float(l9["causal_total"].abs().max()),
        "mean_conditional_effect": float(l9["conditional_effect"].mean()),
        "mean_conditional_gain": float(l9["conditional_gain"].mean()),
        "score_invariant_pass": float(l9["delta_attention"].abs().max()) <= 1e-5
        and float(l9["fali_target"].abs().max()) <= 1e-5,
    }
    scope_md = [
        "# L9H7 scope-boundary report",
        "",
        f"- examples: {scope['n']} (4×200)",
        f"- max |delta_attention|: {scope['max_abs_delta_attention']:.3e}",
        f"- max |delta_target_score|: {scope['max_abs_delta_target_score']:.3e}",
        f"- max |fali_target|: {scope['max_abs_fali_target']:.3e}",
        f"- max |fali_competing|: {scope['max_abs_fali_competing']:.3e}",
        f"- max |causal_total|: {scope['max_abs_causal_total']:.3e}",
        f"- mean conditional_effect: {scope['mean_conditional_effect']:.4f}",
        f"- mean conditional_gain: {scope['mean_conditional_gain']:.4f}",
        f"- score/FALI invariant PASS (≤1e-5): {scope['score_invariant_pass']}",
        "",
        "Conclusion: FALI diagnoses attention-score adaptation; positive functional "
        "recruitment without score change is outside FALI's scope.",
    ]
    (out / "L9H7_scope_report.md").write_text("\n".join(scope_md) + "\n")

    # L10H7 control
    h7 = head_means("L10H7")
    ctrl_md = [
        "# L10H7 suppressor-control report",
        "",
        f"- mean conditional_effect: {h7['mean_conditional_effect']:.4f}",
        f"- mean conditional_gain: {h7['mean_conditional_gain']:.4f}",
        f"- mean delta_attention: {h7['mean_delta_attention']:.4f}",
        f"- mean fali_target: {h7['mean_fali_target']:.4f}",
        f"- mean causal_total: {h7['mean_causal_total']:.4f}",
        f"- mean causal_target: {h7['mean_causal_target']:.4f}",
        "",
        "Question: strong adaptive score-row change while remaining functionally "
        "suppressive (conditional_effect < 0)?",
        f"- Answer: conditional_effect={'negative' if h7['mean_conditional_effect'] < 0 else 'non-negative'}; "
        f"|delta_attention|={abs(h7['mean_delta_attention']):.4f}; "
        f"|causal_total|={abs(h7['mean_causal_total']):.4f}.",
    ]
    (out / "L10H7_control_report.md").write_text("\n".join(ctrl_md) + "\n")
    return {"l10h2_vs_l10h10": (h2, h10), "l9h7": scope, "l10h7": h7}


def make_plots(hs: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # Use seed-mean head points for clarity
    agg = (
        hs.groupby("head_name", sort=False)
        .mean(numeric_only=True)
        .reset_index()
    )

    def scatter(x, y, path, xlabel, ylabel):
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(agg[x], agg[y], s=60, c="#1f4e79")
        for _, r in agg.iterrows():
            ax.annotate(r["head_name"], (r[x], r[y]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_title(f"{ylabel} vs {xlabel} (head means across seeds)")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    scatter(
        "delta_attention",
        "conditional_gain",
        out / "delta_attention_vs_conditional_gain.png",
        "delta_attention",
        "conditional_gain",
    )
    scatter(
        "fali_target",
        "conditional_gain",
        out / "fali_target_vs_conditional_gain.png",
        "fali_target",
        "conditional_gain",
    )
    scatter(
        "fali_target",
        "causal_target",
        out / "fali_target_vs_causal_target.png",
        "fali_target (attn prob units)",
        "causal_target (IO−S units)",
    )
    scatter(
        "fali_competing",
        "causal_competing",
        out / "fali_competing_vs_causal_competing.png",
        "fali_competing (attn prob units)",
        "causal_competing (IO−S units)",
    )
    scatter(
        "causal_total",
        "pattern_clean_patch_effect",
        out / "causal_total_vs_pattern_patch.png",
        "causal_total",
        "pattern_clean_patch_effect",
    )


def classify_result(
    fali_causal: pd.DataFrame,
    advantage: pd.DataFrame,
    scope: Dict,
) -> Tuple[str, List[str]]:
    """
    Predefined rules A/B/C — no post-hoc threshold inventing beyond documented checks.
    """
    notes = []
    # Causal correspondence: across-head seedmean pearson for target/competing
    ac = fali_causal[fali_causal["level"] == "across_head_seedmean"]
    rt = ac[ac["pair"] == "target"]["pearson_r"].iloc[0]
    rc = ac[ac["pair"] == "competing"]["pearson_r"].iloc[0]
    tracks_causal = (
        rt is not None
        and rc is not None
        and float(rt) > 0.5
        and float(rc) > 0.5
    )
    notes.append(f"across-head seedmean pearson fali↔causal target={rt}, competing={rc}")

    # Predictive: FALI improves MAE (lower) and/or R² (higher) vs baselines on average
    # Require reproducible improvement: mean delta_r2 > 0 and mean delta_mae < 0
    # for majority of (cv, target, baseline) groups, with non-tiny effect.
    improvs = []
    for (cv, target, baseline), sub in advantage.groupby(
        ["cv", "target", "baseline"], sort=False
    ):
        mean_dr2 = float(sub["delta_r2_fali_minus_base"].mean())
        mean_dmae = float(sub["delta_mae_fali_minus_base"].mean())
        # tiny: |ΔR²|<0.01 and |ΔMAE|<0.005 → ignore as non-reproducible value-add
        tiny = abs(mean_dr2) < 0.01 and abs(mean_dmae) < 0.005
        better = (mean_dr2 > 0.01) or (mean_dmae < -0.005)
        worse = (mean_dr2 < -0.01) or (mean_dmae > 0.005)
        improvs.append(
            {
                "cv": cv,
                "target": target,
                "baseline": baseline,
                "mean_dr2": mean_dr2,
                "mean_dmae": mean_dmae,
                "tiny": tiny,
                "better": better and not tiny,
                "worse": worse and not tiny,
            }
        )
    n_better = sum(1 for x in improvs if x["better"])
    n_worse = sum(1 for x in improvs if x["worse"])
    n_tiny = sum(1 for x in improvs if x["tiny"])
    notes.append(
        f"predictive groups better={n_better} worse={n_worse} tiny={n_tiny} / {len(improvs)}"
    )
    adds_value = n_better >= max(1, len(improvs) // 2) and n_better > n_worse

    if tracks_causal and adds_value:
        return "STRONG", notes
    if tracks_causal and not adds_value:
        return "NARROW", notes
    if not tracks_causal and not adds_value:
        return "PIVOT", notes
    # tracks poorly but somehow predicts — still not STRONG; treat as PIVOT-leaning NARROW
    return "PIVOT", notes + ["causal tracking failed despite some predictive signal"]


def write_scientific_summary(
    out: Path,
    verdict: str,
    notes: List[str],
    fali_causal: pd.DataFrame,
    advantage: pd.DataFrame,
    contrasts: Dict,
    cv_summary: pd.DataFrame,
) -> None:
    ac = fali_causal[fali_causal["level"] == "across_head_seedmean"]
    lines = [
        "# Phase 2B/3B scientific summary",
        "",
        f"**Predefined verdict: {verdict}**",
        "",
        "## Notes",
    ]
    for n in notes:
        lines.append(f"- {n}")
    lines += ["", "## FALI ↔ causal (across-head, seed-mean heads)"]
    lines.append(ac.to_string(index=False))
    lines += ["", "## Predictive CV summary (mean across folds)"]
    lines.append(cv_summary.to_string(index=False))
    lines += ["", "## FALI vs baselines (mean fold deltas)"]
    adv_sum = (
        advantage.groupby(["cv", "target", "baseline"], sort=False)[
            ["delta_r2_fali_minus_base", "delta_mae_fali_minus_base"]
        ]
        .mean()
        .reset_index()
    )
    lines.append(adv_sum.to_string(index=False))
    lines += [
        "",
        "## L9H7 scope",
        json.dumps(contrasts["l9h7"], indent=2),
        "",
        "## Rules reminder",
        "- STRONG: tracks causal AND improves held-out functional prediction",
        "- NARROW: tracks causal score mediation but does not beat simpler baselines for recruitment",
        "- PIVOT: neither",
    ]
    (out / "phase2_phase3_scientific_summary.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full-root", default="results/revision_phase2/full")
    p.add_argument("--phase1-root", default="results/revision_phase1/full")
    p.add_argument("--out-dir", default="results/revision_phase2/full/aggregate")
    p.add_argument("--boot-seed", type=int, default=42)
    args = p.parse_args()

    full_root = REPO / args.full_root if not Path(args.full_root).is_absolute() else Path(args.full_root)
    phase1_root = (
        REPO / args.phase1_root if not Path(args.phase1_root).is_absolute() else Path(args.phase1_root)
    )
    out = REPO / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plots = out / "plots"
    plots.mkdir(exist_ok=True)

    print("Merging Phase-1 and Phase-2...")
    df = merge_all(full_root, phase1_root)
    df.to_csv(out / "phase1_phase2_merged_raw.csv", index=False)
    print(f"Merged rows: {len(df)}")

    print("Head-seed summaries...")
    hs = head_seed_summary(df)
    hs.to_csv(out / "head_seed_summary.csv", index=False)
    agg = head_aggregate_summary(hs)
    agg.to_csv(out / "head_aggregate_summary.csv", index=False)

    print("Bootstrap CIs (10k stratified)...")
    boot = make_bootstrap(df, base_seed=args.boot_seed)
    boot.to_csv(out / "bootstrap_summary.csv", index=False)

    print("FALI vs causal / function correlations...")
    fc = fali_vs_causal(df, hs)
    fc.to_csv(out / "fali_vs_causal_correlations.csv", index=False)
    ff = fali_vs_function(hs)
    ff.to_csv(out / "fali_vs_function_correlations.csv", index=False)

    print("Predictive CV...")
    folds, cv_sum, advantage = predictive_cv(df)
    folds.to_csv(out / "predictive_cv_fold_results.csv", index=False)
    cv_sum.to_csv(out / "predictive_cv_summary.csv", index=False)
    advantage.to_csv(out / "predictive_fali_advantage.csv", index=False)

    print("Causal baseline comparison...")
    pair_df, corr_df = causal_baseline_comparison(hs)
    pair_df.to_csv(out / "causal_baseline_comparison.csv", index=False)
    corr_df.to_csv(out / "causal_baseline_correlations.csv", index=False)

    print("Template summary...")
    ts = template_summary(df)
    ts.to_csv(out / "template_summary.csv", index=False)

    print("Contrast reports...")
    contrasts = write_contrast_reports(df, hs, out)

    print("Plots...")
    make_plots(hs, plots)

    verdict, notes = classify_result(fc, advantage, contrasts)
    write_scientific_summary(out, verdict, notes, fc, advantage, contrasts, cv_sum)

    meta = {
        "n_rows_merged": len(df),
        "seeds": SEEDS,
        "ridge_alpha": RIDGE_ALPHA,
        "zero_tol_sign_agree": ZERO_TOL,
        "n_bootstrap": N_BOOT,
        "verdict": verdict,
        "notes": notes,
        "reporting_panel": REPORTING_PANEL,
    }
    (out / "aggregate_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"VERDICT: {verdict}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
