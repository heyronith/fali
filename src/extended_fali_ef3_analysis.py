"""
EF-3 two-template scientific analysis — bootstrap, hypotheses, GO/STOP.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from revision_stats import stable_sha256_seed

EF3_SEEDS: Tuple[int, ...] = (42, 1, 8, 22)
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED_TAG = "ef3-two-template-v1"
H1_OWEN_TOL = 1e-6
H1_CONTEXT_TOL = 1e-6
RO_DISCOVERY_FRAC = 0.10

PRIMARY_METRICS = (
    "conditional_gain",
    "phi_T",
    "phi_R",
    "phi_O",
    "phi_V",
    "phi_C",
    "Phi_SCORE",
    "intact_effect",
    "conditional_effect",
)

H2_COMPONENTS = ("conditional_gain", "Phi_SCORE", "phi_T")
H3_COMPONENTS = ("Phi_SCORE", "phi_C", "conditional_gain", "phi_T")
H4_HEADS = ("L10H2", "L10H10")
RO_HEADS = ("L10H2", "L10H10")
H1_HEADS = ("L9H7", "L9H0")


def balanced_bootstrap_means(
    values: np.ndarray,
    seeds: np.ndarray,
    templates: np.ndarray,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed_tag: str = BOOTSTRAP_SEED_TAG,
) -> Tuple[np.ndarray, float]:
    """
    EF-3 primary bootstrap: resample within seed×template cells, equal A/B and seed weights.
    """
    values = np.asarray(values, dtype=np.float64)
    seeds = np.asarray(seeds)
    templates = np.asarray(templates)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        rng = np.random.default_rng(stable_sha256_seed(seed_tag, b))
        seed_means: List[float] = []
        for s in EF3_SEEDS:
            tmpl_means: List[float] = []
            for t in ("A", "B"):
                idx = np.where((seeds == s) & (templates == t))[0]
                if idx.size == 0:
                    raise ValueError(f"Empty stratum seed={s} template={t}")
                draw = rng.choice(idx, size=idx.size, replace=True)
                tmpl_means.append(float(values[draw].mean()))
            seed_means.append(float(np.mean(tmpl_means)))
        boot[b] = float(np.mean(seed_means))
    point = float(np.mean(boot))
    return boot, point


def bootstrap_ci(boot: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    return float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def seed_balanced_means(
    values: np.ndarray,
    seeds: np.ndarray,
    templates: np.ndarray,
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for s in EF3_SEEDS:
        means = []
        for t in ("A", "B"):
            idx = np.where((seeds == s) & (templates == t))[0]
            means.append(float(values[idx].mean()))
        out[int(s)] = float(np.mean(means))
    return out


def direction_support(
    values: np.ndarray,
    seeds: np.ndarray,
    templates: np.ndarray,
    direction: str,
) -> Dict:
    boot, point = balanced_bootstrap_means(values, seeds, templates)
    lo, hi = bootstrap_ci(boot)
    seed_means = seed_balanced_means(values, seeds, templates)
    if direction == "positive":
        ci_ok = lo > 0 and hi > 0
        seeds_ok = all(v > 0 for v in seed_means.values())
    elif direction == "negative":
        ci_ok = hi < 0 and lo < 0
        seeds_ok = all(v < 0 for v in seed_means.values())
    else:
        raise ValueError(direction)
    return {
        "direction": direction,
        "mean": point,
        "ci_low": lo,
        "ci_high": hi,
        "seed_means": seed_means,
        "ci_support": ci_ok,
        "seed_support": seeds_ok,
        "supported": bool(ci_ok and seeds_ok),
    }


def ro_discovery_eval(
    values: np.ndarray,
    seeds: np.ndarray,
    templates: np.ndarray,
    phi_score_mean: float,
    channel: str,
) -> Dict:
    boot, point = balanced_bootstrap_means(values, seeds, templates)
    lo, hi = bootstrap_ci(boot)
    seed_means = seed_balanced_means(values, seeds, templates)
    threshold = RO_DISCOVERY_FRAC * abs(phi_score_mean)
    same_sign = len(set(np.sign(list(seed_means.values())))) == 1
    nonzero_sign = list(seed_means.values())[0] != 0 if same_sign else False
    ci_excludes_zero = lo > 0 or hi < 0
    magnitude_ok = abs(point) >= threshold
    passed = bool(ci_excludes_zero and same_sign and nonzero_sign and magnitude_ok)
    return {
        "channel": channel,
        "mean": point,
        "ci_low": lo,
        "ci_high": hi,
        "seed_means": seed_means,
        "threshold": threshold,
        "ci_excludes_zero": ci_excludes_zero,
        "seed_same_sign": same_sign,
        "magnitude_ok": magnitude_ok,
        "discovery_candidate": passed,
    }


def eval_h1(df: pd.DataFrame) -> Dict:
    results = {}
    for head in H1_HEADS:
        sub = df[df["head_name"] == head]
        results[head] = {
            "max_abs_phi_T": float(sub["phi_T"].abs().max()),
            "max_abs_phi_R": float(sub["phi_R"].abs().max()),
            "max_abs_phi_O": float(sub["phi_O"].abs().max()),
            "max_abs_phi_V": float(sub["phi_V"].abs().max()),
            "max_context_residual": float((sub["phi_C"] - sub["conditional_gain"]).abs().max()),
            "architectural_pass": bool(
                sub["phi_T"].abs().max() <= H1_OWEN_TOL
                and sub["phi_R"].abs().max() <= H1_OWEN_TOL
                and sub["phi_O"].abs().max() <= H1_OWEN_TOL
                and sub["phi_V"].abs().max() <= H1_OWEN_TOL
                and (sub["phi_C"] - sub["conditional_gain"]).abs().max() <= H1_CONTEXT_TOL
            ),
        }
        boot, point = balanced_bootstrap_means(
            sub["phi_C"].values, sub["seed"].values, sub["template_id"].values
        )
        lo, hi = bootstrap_ci(boot)
        results[head]["phi_C_mean"] = point
        results[head]["phi_C_ci"] = [lo, hi]
        results[head]["phi_C_seed_means"] = seed_balanced_means(
            sub["phi_C"].values, sub["seed"].values, sub["template_id"].values
        )
    results["L9H7_context_only_pass"] = results["L9H7"]["architectural_pass"]
    return results


def eval_h2_h3(df: pd.DataFrame, head: str, components: Sequence[str], directions: Dict[str, str]) -> Dict:
    sub = df[df["head_name"] == head]
    out = {"head": head, "components": {}}
    for comp in components:
        out["components"][comp] = direction_support(
            sub[comp].values, sub["seed"].values, sub["template_id"].values, directions[comp]
        )
    return out


def eval_h4(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for head in H4_HEADS:
        sub = df[df["head_name"] == head]
        row = {"head_name": head}
        for col in [
            "clean_target_attn",
            "damaged_target_attn",
            "delta_target_attn",
            "desc_phi_target",
            "desc_phi_competing",
            "phi_T",
            "phi_R",
            "phi_O",
            "Phi_SCORE",
            "phi_V",
            "phi_C",
            "conditional_gain",
        ]:
            boot, point = balanced_bootstrap_means(
                sub[col].values, sub["seed"].values, sub["template_id"].values
            )
            lo, hi = bootstrap_ci(boot)
            row[f"{col}_mean"] = point
            row[f"{col}_ci_low"] = lo
            row[f"{col}_ci_high"] = hi
            row[f"{col}_seed_means"] = json.dumps(
                seed_balanced_means(sub[col].values, sub["seed"].values, sub["template_id"].values)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def eval_all_heads(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for head in sorted(df["head_name"].unique()):
        sub = df[df["head_name"] == head]
        row = {"head_name": head}
        for metric in PRIMARY_METRICS:
            boot, point = balanced_bootstrap_means(
                sub[metric].values, sub["seed"].values, sub["template_id"].values
            )
            lo, hi = bootstrap_ci(boot)
            row[f"{metric}_mean"] = point
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
            row[f"{metric}_frac_positive"] = float((sub[metric] > 0).mean())
            row[f"{metric}_seed_means"] = json.dumps(
                seed_balanced_means(sub[metric].values, sub["seed"].values, sub["template_id"].values)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def descriptive_vs_causal_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for head in sorted(df["head_name"].unique()):
        sub = df[df["head_name"] == head]
        if len(sub) < 3:
            continue
        pr_t, _ = stats.pearsonr(sub["desc_phi_target"], sub["phi_T"])
        sr_t, _ = stats.spearmanr(sub["desc_phi_target"], sub["phi_T"])
        pr_r, _ = stats.pearsonr(sub["desc_phi_competing"], sub["phi_R"])
        sr_r, _ = stats.spearmanr(sub["desc_phi_competing"], sub["phi_R"])
        rows.append(
            {
                "head_name": head,
                "pearson_desc_target_vs_phi_T": float(pr_t),
                "spearman_desc_target_vs_phi_T": float(sr_t),
                "pearson_desc_competing_vs_phi_R": float(pr_r),
                "spearman_desc_competing_vs_phi_R": float(sr_r),
            }
        )
    head_means = df.groupby("head_name")[["desc_phi_target", "phi_T", "desc_phi_competing", "phi_R"]].mean()
    if len(head_means) >= 3:
        rows.append(
            {
                "head_name": "__across_head__",
                "pearson_desc_target_vs_phi_T": float(head_means["desc_phi_target"].corr(head_means["phi_T"])),
                "spearman_desc_target_vs_phi_T": float(
                    stats.spearmanr(head_means["desc_phi_target"], head_means["phi_T"]).statistic
                ),
                "pearson_desc_competing_vs_phi_R": float(
                    head_means["desc_phi_competing"].corr(head_means["phi_R"])
                ),
                "spearman_desc_competing_vs_phi_R": float(
                    stats.spearmanr(head_means["desc_phi_competing"], head_means["phi_R"]).statistic
                ),
            }
        )
    return pd.DataFrame(rows)


def go_stop_decision(
    validation_pass: bool,
    h1: Dict,
    ro_discovery: Dict,
) -> Dict:
    h1_ok = h1.get("L9H7_context_only_pass", False)
    stable_channels = []
    for head in RO_HEADS:
        for ch in ("phi_R", "phi_O"):
            key = f"{head}_{ch}"
            if ro_discovery.get(key, {}).get("discovery_candidate"):
                stable_channels.append(key)
    criterion_c = len(stable_channels) > 0
    decision = "GO" if (validation_pass and h1_ok and criterion_c) else "STOP"
    return {
        "decision": decision,
        "validation_pass": validation_pass,
        "h1_l9h7_architectural_pass": h1_ok,
        "stable_score_channels": stable_channels,
        "criterion_c_pass": criterion_c,
    }


def load_merged_attributions(aggregate_dir: Path, repo_root: Path) -> pd.DataFrame:
    merged = aggregate_dir / "attribution_merged_raw.csv"
    if merged.exists():
        return pd.read_csv(merged)
    parts = []
    for seed in EF3_SEEDS:
        p = repo_root / f"results/extended_fali/ef3_two_template/seed_{seed}/attribution_raw.csv"
        parts.append(pd.read_csv(p))
    return pd.concat(parts, ignore_index=True)


def run_analysis(repo_root: Path, aggregate_dir: Path) -> Dict:
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    df = load_merged_attributions(aggregate_dir, repo_root)
    df.to_csv(aggregate_dir / "attribution_merged_raw.csv", index=False)

    seed_rows = []
    for seed in EF3_SEEDS:
        for head in sorted(df["head_name"].unique()):
            sub = df[(df["seed"] == seed) & (df["head_name"] == head)]
            row = {"seed": seed, "head_name": head, "n": len(sub)}
            for metric in PRIMARY_METRICS:
                row[f"{metric}_mean"] = float(sub[metric].mean())
            seed_rows.append(row)
    pd.DataFrame(seed_rows).to_csv(aggregate_dir / "seed_means.csv", index=False)

    head_means = eval_all_heads(df)
    head_means.to_csv(aggregate_dir / "head_means.csv", index=False)

    boot_rows = []
    for head in sorted(df["head_name"].unique()):
        sub = df[df["head_name"] == head]
        for metric in PRIMARY_METRICS:
            boot, point = balanced_bootstrap_means(
                sub[metric].values, sub["seed"].values, sub["template_id"].values
            )
            lo, hi = bootstrap_ci(boot)
            boot_rows.append(
                {
                    "head_name": head,
                    "metric": metric,
                    "mean": point,
                    "ci_low": lo,
                    "ci_high": hi,
                    "seed_means": json.dumps(
                        seed_balanced_means(sub[metric].values, sub["seed"].values, sub["template_id"].values)
                    ),
                }
            )
    pd.DataFrame(boot_rows).to_csv(aggregate_dir / "bootstrap_ci.csv", index=False)

    h1 = eval_h1(df)
    h2 = eval_h2_h3(
        df,
        "L10H2",
        H2_COMPONENTS,
        {"conditional_gain": "positive", "Phi_SCORE": "positive", "phi_T": "positive"},
    )
    h3 = eval_h2_h3(
        df,
        "L10H10",
        H3_COMPONENTS,
        {"Phi_SCORE": "positive", "phi_C": "negative", "conditional_gain": "negative", "phi_T": "positive"},
    )
    h4 = eval_h4(df)
    h4.to_csv(aggregate_dir / "l10h2_vs_l10h10.csv", index=False)

    ro = {}
    for head in RO_HEADS:
        sub = df[df["head_name"] == head]
        phi_score_mean = float(
            balanced_bootstrap_means(
                sub["Phi_SCORE"].values, sub["seed"].values, sub["template_id"].values
            )[1]
        )
        for ch in ("phi_R", "phi_O"):
            ro[f"{head}_{ch}"] = ro_discovery_eval(
                sub[ch].values, sub["seed"].values, sub["template_id"].values, phi_score_mean, ch
            )

    corr = descriptive_vs_causal_correlations(df)
    corr.to_csv(aggregate_dir / "descriptive_vs_causal_correlations.csv", index=False)

    val_summary_path = aggregate_dir / "validation_summary.json"
    validation_pass = True
    if val_summary_path.exists():
        validation_pass = json.loads(val_summary_path.read_text()).get("overall_pass", False)

    go_stop = go_stop_decision(validation_pass, h1, ro)

    hypothesis = {"H1": h1, "H2_L10H2": h2, "H3_L10H10": h3}
    (aggregate_dir / "hypothesis_results.json").write_text(json.dumps(hypothesis, indent=2) + "\n")
    (aggregate_dir / "exploratory_ro_discovery.json").write_text(json.dumps(ro, indent=2) + "\n")

    summary_lines = [
        "# EF-3 Extended FALI scientific summary (two-template mixed, N=200 × 4 seeds)",
        "",
        "This analysis is under the preregistered two-template distribution only.",
        "Seeds are prompt-generation cohorts, not model replicas.",
        "",
        f"GO/STOP decision: **{go_stop['decision']}**",
        "",
        "## H1 (L9H7 architectural context-only)",
        f"- L9H7 architectural pass: {h1['L9H7']['architectural_pass']}",
        "",
        "## H2 L10H2 (confirmatory components reported separately)",
    ]
    for comp, res in h2["components"].items():
        summary_lines.append(
            f"- {comp}: supported={res['supported']} mean={res['mean']:.6f} CI=[{res['ci_low']:.6f}, {res['ci_high']:.6f}]"
        )
    summary_lines.append("")
    summary_lines.append("## H3 L10H10")
    for comp, res in h3["components"].items():
        summary_lines.append(
            f"- {comp}: supported={res['supported']} mean={res['mean']:.6f} CI=[{res['ci_low']:.6f}, {res['ci_high']:.6f}]"
        )
    summary_lines.append("")
    summary_lines.append("## Safety note")
    summary_lines.append(
        "Interpretability-focused; mitigations: preregistration, historical bridge, canonical confirmation gate."
    )
    (aggregate_dir / "scientific_summary.md").write_text("\n".join(summary_lines) + "\n")

    analysis_config = {
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed_tag": BOOTSTRAP_SEED_TAG,
        "seeds": list(EF3_SEEDS),
        "estimand": "balanced mean over 4 seeds × 2 templates × 100 prompts",
    }
    (aggregate_dir / "analysis_config.json").write_text(json.dumps(analysis_config, indent=2) + "\n")

    return {"hypothesis": hypothesis, "go_stop": go_stop, "ro_discovery": ro}
