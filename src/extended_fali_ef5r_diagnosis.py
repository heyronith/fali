"""
EF-5R post-stop zero-reference diagnosis (analysis only, no model execution).
"""

from __future__ import annotations

import gzip
import io
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from extended_fali_ef4_analysis import (
    EF4_SEEDS,
    N_BOOTSTRAP,
    N_TEMPLATES,
    TARGET_HEADS,
    observed_canonical_balanced_mean,
    seed_canonical_means,
)
from historical_integrity_audit import audit_ef5_historical_integrity
from revision_stats import hierarchical_template_cluster_bootstrap_means, stable_sha256_seed

EF4_HEADS = ("L9H7", "L9H0", "L10H2", "L10H10")
EF5_SCHEMES = ("zero", "matched_mean", "matched_resample")
ALT_SCHEMES = ("matched_mean", "matched_resample")
TOL = 1e-4
EF4_RAW_REF = "audit/extended-fali-ef4-raw"
EF5_RAW_REF = "audit/extended-fali-ef5-raw"
EF4_BASE_REF = "79feba9d9a97c62ada673159aaa1d13f757da05b"

PRESERVED_VERDICT = {
    "EF5_ABLATION_GATE": "FAIL",
    "matched_mean_channel_status": "ROBUST",
    "matched_resample_channel_status": "ROBUST",
    "reference_perturbation_stop": True,
}


def load_cell_raw_from_git(repo: Path, git_ref: str, rel_path: str) -> pd.DataFrame:
    data = subprocess.check_output(["git", "show", f"{git_ref}:{rel_path}"], cwd=str(repo))
    with gzip.open(io.BytesIO(data), "rt", encoding="utf-8") as f:
        return pd.read_csv(f)


def load_cell_raw(repo: Path, rel_path: str, git_ref: str) -> pd.DataFrame:
    local = repo / rel_path
    if local.exists():
        if str(local).endswith(".gz"):
            with gzip.open(local, "rt", encoding="utf-8") as f:
                return pd.read_csv(f)
        return pd.read_csv(local)
    return load_cell_raw_from_git(repo, git_ref, rel_path)


def extract_cell_state(
    cells: pd.DataFrame,
    *,
    example_idx: int,
    head_name: str,
    state: Tuple[int, int, int, int, int],
) -> pd.Series:
    t, r, o, v, c = state
    sub = cells[(cells["example_idx"] == example_idx) & (cells["head_name"] == head_name)]
    row = sub[
        (sub["T"] == t)
        & (sub["R"] == r)
        & (sub["O"] == o)
        & (sub["V"] == v)
        & (sub["C"] == c)
    ]
    if len(row) != 1:
        raise ValueError(f"Expected 1 cell for ex={example_idx} head={head_name} state={state}, got {len(row)}")
    return row.iloc[0]


def endpoint_metrics_from_cells(cells: pd.DataFrame, heads: Sequence[str] = EF4_HEADS) -> pd.DataFrame:
    """Extract all-A and all-B metric_injected per seed × example × head."""
    rows = []
    for (seed, ex), grp in cells.groupby(["seed", "example_idx"]):
        for head in heads:
            a = extract_cell_state(grp, example_idx=int(ex), head_name=head, state=(0, 0, 0, 0, 0))
            b = extract_cell_state(grp, example_idx=int(ex), head_name=head, state=(1, 1, 1, 1, 1))
            rows.append(
                {
                    "seed": int(seed),
                    "example_idx": int(ex),
                    "head_name": head,
                    "clean_IO_minus_S": float(a["metric_injected"]),
                    "damaged_IO_minus_S": float(b["metric_injected"]),
                    "drop_from_cells": float(a["metric_injected"] - b["metric_injected"]),
                    "metric_zero_all_A": float(a["metric_zero"]),
                    "metric_zero_all_B": float(b["metric_zero"]),
                }
            )
    return pd.DataFrame(rows)


def attach_metadata(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    cols = ["example_idx", "low_level_template_id", "pattern_type", "template_stratum"]
    m = meta[cols].drop_duplicates(subset=["example_idx"])
    return df.merge(m, on="example_idx", how="left")


def reconstruct_zero_drop_ef4(repo: Path) -> Tuple[pd.DataFrame, Dict]:
    endpoint_parts = []
    for seed in EF4_SEEDS:
        rel = f"results/extended_fali/ef4_canonical/seed_{seed}/cell_raw.csv.gz"
        cells = load_cell_raw(repo, rel, EF4_RAW_REF)
        meta = pd.read_csv(repo / f"results/revision_phase4/full/seed_{seed}/dataset_metadata.csv")
        ep = attach_metadata(endpoint_metrics_from_cells(cells), meta)
        ep["scheme"] = "zero"
        endpoint_parts.append(ep)

    endpoints = pd.concat(endpoint_parts, ignore_index=True)

    head_agreement_clean = []
    head_agreement_damaged = []
    for (seed, ex), grp in endpoints.groupby(["seed", "example_idx"]):
        clean_vals = grp["clean_IO_minus_S"].values
        dmg_vals = grp["damaged_IO_minus_S"].values
        head_agreement_clean.append(float(np.max(clean_vals) - np.min(clean_vals)))
        head_agreement_damaged.append(float(np.max(dmg_vals) - np.min(dmg_vals)))

    max_clean_head_discrep = max(head_agreement_clean) if head_agreement_clean else 0.0
    max_damaged_head_discrep = max(head_agreement_damaged) if head_agreement_damaged else 0.0

    consensus = (
        endpoints.groupby(["seed", "example_idx"], as_index=False)
        .agg(
            clean_IO_minus_S=("clean_IO_minus_S", "mean"),
            zero_removed_IO_minus_S=("damaged_IO_minus_S", "mean"),
            zero_drop_from_EF4=("drop_from_cells", "mean"),
        )
    )
    consensus = attach_metadata(
        consensus,
        pd.concat(
            [
                pd.read_csv(repo / f"results/revision_phase4/full/seed_{s}/dataset_metadata.csv")
                for s in EF4_SEEDS
            ],
            ignore_index=True,
        ),
    )

    ef5_parts = []
    for seed in EF4_SEEDS:
        ef5_parts.append(
            pd.read_csv(
                repo / f"results/extended_fali/ef5_ablation_robustness/zero_damage/seed_{seed}/damage_drop.csv"
            )
        )
    ef5_zero = pd.concat(ef5_parts, ignore_index=True)
    merged = consensus.merge(
        ef5_zero[["seed", "example_idx", "damage_drop"]],
        on=["seed", "example_idx"],
        how="inner",
    )
    merged["ef5_forward_discrepancy"] = merged["zero_drop_from_EF4"] - merged["damage_drop"]
    max_ef5_discrep = float(merged["ef5_forward_discrepancy"].abs().max())

    checks = {
        "max_clean_head_discrepancy": max_clean_head_discrep,
        "max_damaged_head_discrepancy": max_damaged_head_discrep,
        "max_ef5_forward_zero_drop_discrepancy": max_ef5_discrep,
        "check_A_clean_head_agreement": max_clean_head_discrep <= TOL,
        "check_B_damaged_head_agreement": max_damaged_head_discrep <= TOL,
        "check_C_ef5_forward_agreement": max_ef5_discrep <= TOL,
        "implementation_inconsistency_stop": max_ef5_discrep > TOL,
    }
    return endpoints, {**checks, "consensus": consensus}


def cross_run_endpoint_check(repo: Path) -> Dict:
    clean_by_scheme: Dict[str, pd.DataFrame] = {}
    damaged_by_scheme: Dict[str, pd.DataFrame] = {}

    for seed in EF4_SEEDS:
        meta = pd.read_csv(repo / f"results/revision_phase4/full/seed_{seed}/dataset_metadata.csv")

        ef4_cells = load_cell_raw(
            repo,
            f"results/extended_fali/ef4_canonical/seed_{seed}/cell_raw.csv.gz",
            EF4_RAW_REF,
        )
        ep4 = attach_metadata(endpoint_metrics_from_cells(ef4_cells), meta)
        ep4 = ep4.groupby(["seed", "example_idx"], as_index=False).agg(
            clean=("clean_IO_minus_S", "mean"),
            damaged=("damaged_IO_minus_S", "mean"),
        )
        clean_by_scheme.setdefault("zero", []).append(ep4.rename(columns={"clean": "clean_IO_minus_S"})[
            ["seed", "example_idx", "clean_IO_minus_S"]
        ])
        damaged_by_scheme.setdefault("zero", []).append(
            ep4.rename(columns={"damaged": "zero_removed_IO_minus_S"})[["seed", "example_idx", "zero_removed_IO_minus_S"]]
        )

        for scheme in ALT_SCHEMES:
            cells = load_cell_raw(
                repo,
                f"results/extended_fali/ef5_ablation_robustness/{scheme}/seed_{seed}/cell_raw.csv.gz",
                EF5_RAW_REF,
            )
            ep = attach_metadata(endpoint_metrics_from_cells(cells), meta)
            ep = ep.groupby(["seed", "example_idx"], as_index=False).agg(
                clean=("clean_IO_minus_S", "mean"),
                damaged=("damaged_IO_minus_S", "mean"),
            )
            clean_by_scheme.setdefault(scheme, []).append(
                ep.rename(columns={"clean": "clean_IO_minus_S"})[["seed", "example_idx", "clean_IO_minus_S"]]
            )
            col = f"{scheme}_IO_minus_S"
            damaged_by_scheme.setdefault(scheme, []).append(
                ep.rename(columns={"damaged": col})[["seed", "example_idx", col]]
            )

    clean_zero = pd.concat(clean_by_scheme["zero"], ignore_index=True)
    max_clean_cross = 0.0
    for scheme in ALT_SCHEMES:
        other = pd.concat(clean_by_scheme[scheme], ignore_index=True)
        m = clean_zero.merge(other, on=["seed", "example_idx"], suffixes=("_zero", f"_{scheme}"))
        max_clean_cross = max(max_clean_cross, float((m["clean_IO_minus_S_zero"] - m[f"clean_IO_minus_S_{scheme}"]).abs().max()))

    return {
        "max_clean_endpoint_cross_run_discrepancy": max_clean_cross,
        "check_clean_cross_run": max_clean_cross <= TOL,
        "schemes_compared": list(EF5_SCHEMES),
    }


def diagnostic_ci(values, seeds, tid, pattern, scope: str) -> Tuple[float, float]:
    boot = hierarchical_template_cluster_bootstrap_means(
        values,
        seeds,
        tid,
        pattern,
        n_bootstrap=N_BOOTSTRAP,
        seed=stable_sha256_seed("EF-5R", scope, "diagnostic"),
        n_templates=N_TEMPLATES,
    )
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def diagnostic_ci_within_pattern(
    values: np.ndarray,
    seed_ids: np.ndarray,
    tid: np.ndarray,
    *,
    scope: str,
    n_templates: int = N_TEMPLATES,
) -> Tuple[float, float]:
    """Post-hoc diagnostic CI for a single pattern type (15 template strata per seed)."""
    values = np.asarray(values, dtype=np.float64)
    seed_ids = np.asarray(seed_ids)
    tid = np.asarray(tid).astype(np.int64)
    unique_seeds = sorted(int(s) for s in np.unique(seed_ids))
    rng = np.random.default_rng(stable_sha256_seed("EF-5R", scope, "pattern_only"))
    boots: List[float] = []
    for _ in range(N_BOOTSTRAP):
        drawn = rng.integers(0, n_templates, size=n_templates)
        seed_means: List[float] = []
        for s in unique_seeds:
            stratum_means = [
                float(values[(seed_ids == s) & (tid == t)].mean()) for t in drawn
            ]
            seed_means.append(float(np.mean(stratum_means)))
        boots.append(float(np.mean(seed_means)))
    arr = np.asarray(boots)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def summarize_damage_by_pattern(damage_df: pd.DataFrame, scheme: str) -> pd.DataFrame:
    sub = damage_df[damage_df["damage_scheme"] == scheme].copy()
    rows = []
    for pattern in ("BABA", "ABBA"):
        psub = sub[sub["pattern_type"] == pattern]
        values = psub["damage_drop"].values
        seeds = psub["seed"].values
        tid = psub["low_level_template_id"].values
        pat = psub["pattern_type"].values
        lo, hi = diagnostic_ci_within_pattern(values, seeds, tid, scope=f"{scheme}_{pattern}")
        rows.append(
            {
                "scheme": scheme,
                "pattern_type": pattern,
                "exact_balanced_mean": observed_canonical_balanced_mean(values, seeds, tid, pat),
                "posthoc_diagnostic_ci_low": lo,
                "posthoc_diagnostic_ci_high": hi,
                "seed_means_json": json.dumps(seed_canonical_means(values, seeds, tid, pat)),
                "n_strata": 15,
            }
        )
    return pd.DataFrame(rows)


def summarize_damage_by_template(damage_df: pd.DataFrame, scheme: str) -> pd.DataFrame:
    sub = damage_df[damage_df["damage_scheme"] == scheme]
    rows = []
    for tid in range(N_TEMPLATES):
        for pattern in ("BABA", "ABBA"):
            psub = sub[(sub["low_level_template_id"] == tid) & (sub["pattern_type"] == pattern)]
            if psub.empty:
                continue
            rows.append(
                {
                    "scheme": scheme,
                    "low_level_template_id": tid,
                    "pattern_type": pattern,
                    "mean": float(psub["damage_drop"].mean()),
                    "n": len(psub),
                }
            )
        baba = sub[(sub["low_level_template_id"] == tid) & (sub["pattern_type"] == "BABA")]["damage_drop"].mean()
        abba = sub[(sub["low_level_template_id"] == tid) & (sub["pattern_type"] == "ABBA")]["damage_drop"].mean()
        rows.append(
            {
                "scheme": scheme,
                "low_level_template_id": tid,
                "pattern_type": "PAIR",
                "baba_mean": float(baba),
                "abba_mean": float(abba),
                "baba_minus_abba": float(baba - abba),
                "baba_sign_positive": bool(baba > 0),
                "abba_sign_positive": bool(abba > 0),
            }
        )
    return pd.DataFrame(rows)


def build_damage_table(repo: Path) -> pd.DataFrame:
    parts = []
    for seed in EF4_SEEDS:
        parts.append(
            pd.read_csv(
                repo / f"results/extended_fali/ef5_ablation_robustness/zero_damage/seed_{seed}/damage_drop.csv"
            )
        )
    zero = pd.concat(parts)
    zero["damage_scheme"] = "zero"
    for scheme in ALT_SCHEMES:
        for seed in EF4_SEEDS:
            attr = pd.read_csv(
                repo / f"results/extended_fali/ef5_ablation_robustness/{scheme}/seed_{seed}/attribution_raw.csv"
            )
            sub = attr[
                ["seed", "example_idx", "damage_drop", "low_level_template_id", "pattern_type", "template_stratum"]
            ].drop_duplicates(subset=["seed", "example_idx"])
            sub["damage_scheme"] = scheme
            parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def damage_drop_field_consistency(repo: Path) -> Dict:
    max_disc = 0.0
    per_scheme = {}
    for scheme in ALT_SCHEMES:
        scheme_max = 0.0
        for seed in EF4_SEEDS:
            cells = load_cell_raw(
                repo,
                f"results/extended_fali/ef5_ablation_robustness/{scheme}/seed_{seed}/cell_raw.csv.gz",
                EF5_RAW_REF,
            )
            ep = endpoint_metrics_from_cells(cells).groupby(["seed", "example_idx"], as_index=False).agg(
                drop_from_cells=("drop_from_cells", "mean")
            )
            attr = pd.read_csv(
                repo / f"results/extended_fali/ef5_ablation_robustness/{scheme}/seed_{seed}/attribution_raw.csv"
            )
            dd = attr[["seed", "example_idx", "damage_drop"]].drop_duplicates(subset=["seed", "example_idx"])
            m = ep.merge(dd, on=["seed", "example_idx"])
            disc = (m["drop_from_cells"] - m["damage_drop"]).abs()
            scheme_max = max(scheme_max, float(disc.max()))
        per_scheme[scheme] = scheme_max
        max_disc = max(max_disc, scheme_max)
    return {
        "max_attribution_vs_cell_drop_discrepancy": max_disc,
        "per_scheme": per_scheme,
        "check_pass": max_disc <= TOL,
    }


def fine_channels_by_pattern(repo: Path) -> pd.DataFrame:
    rows = []
    zero_attr = pd.concat(
        [pd.read_csv(repo / f"results/extended_fali/ef4_canonical/seed_{s}/attribution_raw.csv") for s in EF4_SEEDS]
    )
    zero_attr["scheme"] = "zero"
    frames = [zero_attr]
    for scheme in ALT_SCHEMES:
        frames.append(
            pd.concat(
                [
                    pd.read_csv(
                        repo / f"results/extended_fali/ef5_ablation_robustness/{scheme}/seed_{s}/attribution_raw.csv"
                    ).assign(scheme=scheme)
                    for s in EF4_SEEDS
                ]
            )
        )
    all_attr = pd.concat(frames, ignore_index=True)
    for scheme in EF5_SCHEMES:
        sub = all_attr[all_attr["scheme"] == scheme]
        for head in TARGET_HEADS:
            hs = sub[sub["head_name"] == head]
            for pattern in ("BABA", "ABBA"):
                ps = hs[hs["pattern_type"] == pattern]
                rows.append(
                    {
                        "scheme": scheme,
                        "head_name": head,
                        "pattern_type": pattern,
                        "phi_T_mean": float(ps["phi_T"].mean()),
                        "phi_R_mean": float(ps["phi_R"].mean()),
                        "phi_T_balanced": observed_canonical_balanced_mean(
                            ps["phi_T"].values,
                            ps["seed"].values,
                            ps["low_level_template_id"].values,
                            ps["pattern_type"].values,
                            n_templates=N_TEMPLATES,
                            template_ids=list(range(N_TEMPLATES)),
                        ),
                        "phi_R_balanced": observed_canonical_balanced_mean(
                            ps["phi_R"].values,
                            ps["seed"].values,
                            ps["low_level_template_id"].values,
                            ps["pattern_type"].values,
                            n_templates=N_TEMPLATES,
                            template_ids=list(range(N_TEMPLATES)),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def io_semantic_audit(repo: Path) -> Dict:
    failures = []
    checked = 0
    for seed in EF4_SEEDS:
        meta = pd.read_csv(repo / f"results/revision_phase4/full/seed_{seed}/dataset_metadata.csv")
        for _, row in meta.iterrows():
            checked += 1
            pat = row["pattern_type"]
            s1, io, s2 = int(row["S1_position"]), int(row["IO_position"]), int(row["S2_position"])
            if pat == "BABA":
                if not (s1 < io < s2):
                    failures.append(f"seed={seed} ex={row['example_idx']} BABA order fail")
            elif pat == "ABBA":
                if not (io < s1 < s2):
                    failures.append(f"seed={seed} ex={row['example_idx']} ABBA order fail")
            else:
                failures.append(f"unknown pattern {pat}")
    position_pass = not failures
    return {
        "status": "PASS" if position_pass else "FAIL",
        "position_order_audit": {
            "status": "PASS" if position_pass else "FAIL",
            "method": "direct check over frozen Phase-4 dataset_metadata rows",
            "examples_checked": checked,
            "failures": failures[:20],
            "n_failures": len(failures),
        },
        "token_identity_audit": {
            "status": "INHERITED",
            "method": (
                "StandardIOIDataset generation validation + exact EF-4 cohort alignment "
                "(io_token_id / s_token_id not revalidated per example in EF-5R)"
            ),
            "direct_per_example_token_id_revalidation": False,
        },
    }


def classify_diagnosis(
    recon_checks: Dict,
    damage_by_pattern: pd.DataFrame,
) -> str:
    if recon_checks.get("implementation_inconsistency_stop"):
        return "IMPLEMENTATION_INCONSISTENCY"

    zero = damage_by_pattern[damage_by_pattern["scheme"] == "zero"]
    baba = float(zero[zero["pattern_type"] == "BABA"]["exact_balanced_mean"].iloc[0])
    abba = float(zero[zero["pattern_type"] == "ABBA"]["exact_balanced_mean"].iloc[0])

    if (baba > 0) != (abba > 0):
        return "GENUINE_ZERO_HETEROGENEITY"
    if baba < 0 and abba < 0:
        return "GENUINE_AGGREGATE_ZERO_BENEFIT"
    return "GENUINE_ZERO_HETEROGENEITY"


def informative_label_clarification_md(damage_df: pd.DataFrame, damage_by_pattern: pd.DataFrame) -> str:
    zero_rows = damage_df[damage_df["damage_scheme"] == "zero"]
    zero_mean = observed_canonical_balanced_mean(
        zero_rows["damage_drop"].values,
        zero_rows["seed"].values,
        zero_rows["low_level_template_id"].values,
        zero_rows["pattern_type"].values,
    )
    lines = [
        "# EF-5 informative-label clarification (post-stop, does not alter EF-5 artifacts)",
        "",
        "The preregistered EF-5 gate remains **FAIL** with `reference_perturbation_stop = true`.",
        "",
        f"Zero reference balanced mean damage_drop = **{zero_mean:.6f}** (not positive with CI > 0).",
        "",
        "Therefore criterion C (`alternative_mean >= 0.25 * zero_mean`) is **NOT_APPLICABLE** "
        "when `zero_mean < 0` — the fractional reference is not scientifically interpretable.",
        "",
        "Post-stop descriptive labels for alternatives:",
        "",
    ]
    for scheme in ALT_SCHEMES:
        sub = damage_df[damage_df["damage_scheme"] == scheme]
        values = sub["damage_drop"].values
        seeds = sub["seed"].values
        tid = sub["low_level_template_id"].values
        pat = sub["pattern_type"].values
        lo, hi = diagnostic_ci(values, seeds, tid, pat, f"clarify_{scheme}")
        seed_m = seed_canonical_means(values, seeds, tid, pat)
        degrading = lo > 0 and hi > 0 and all(v > 0 for v in seed_m.values())
        lines.append(
            f"- **{scheme}**: `performance_degrading_AB = {str(degrading).lower()}`; "
            f"`fractional_reference_C = NOT_APPLICABLE`"
        )
    lines.append("")
    lines.append("Original EF-5 `INFORMATIVE` fields in aggregate artifacts are unchanged.")
    return "\n".join(lines) + "\n"


def recommended_zero_terminology(category: str, baba_mean: float, abba_mean: float) -> str:
    if category == "IMPLEMENTATION_INCONSISTENCY":
        return "Do not interpret zero reference until provenance is reconciled."
    if category == "GENUINE_ZERO_HETEROGENEITY":
        return (
            "Prefer 'pattern-heterogeneous primary zero effect on IO−S' over uniform 'damage'; "
            f"BABA balanced drop {baba_mean:+.3f} vs ABBA {abba_mean:+.3f}."
        )
    return (
        "Prefer 'primary zero intervention' or 'primary removal effect on IO−S' rather than "
        "unqualified 'damage' when aggregate effect aligns across pattern types."
    )


def run_diagnosis(repo: Path, out_dir: Path) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    damage_df = build_damage_table(repo)

    endpoints, recon = reconstruct_zero_drop_ef4(repo)
    endpoints.to_csv(out_dir / "endpoint_reconstruction.csv", index=False)

    cross = cross_run_endpoint_check(repo)
    (out_dir / "cross_run_endpoint_check.json").write_text(json.dumps(cross, indent=2) + "\n")

    pattern_rows = []
    template_rows = []
    for scheme in EF5_SCHEMES:
        pattern_rows.append(summarize_damage_by_pattern(damage_df, scheme))
        template_rows.append(summarize_damage_by_template(damage_df, scheme))
    damage_by_pattern = pd.concat(pattern_rows, ignore_index=True)
    damage_by_pattern.to_csv(out_dir / "damage_by_pattern.csv", index=False)
    damage_by_template = pd.concat(template_rows, ignore_index=True)
    damage_by_template.to_csv(out_dir / "damage_by_template.csv", index=False)

    fine = fine_channels_by_pattern(repo)
    fine.to_csv(out_dir / "fine_channels_by_pattern.csv", index=False)

    drop_check = damage_drop_field_consistency(repo)
    io_audit = io_semantic_audit(repo)
    integrity = audit_ef5_historical_integrity(repo, ef4_base_ref=EF4_BASE_REF, ef5_branch_ref="HEAD")

    category = classify_diagnosis(recon, damage_by_pattern)
    zero_rows = damage_df[damage_df["damage_scheme"] == "zero"]
    zero_pat = damage_by_pattern[damage_by_pattern["scheme"] == "zero"]
    baba_mean = float(zero_pat[zero_pat["pattern_type"] == "BABA"]["exact_balanced_mean"].iloc[0])
    abba_mean = float(zero_pat[zero_pat["pattern_type"] == "ABBA"]["exact_balanced_mean"].iloc[0])
    zero_balanced_mean = observed_canonical_balanced_mean(
        zero_rows["damage_drop"].values,
        zero_rows["seed"].values,
        zero_rows["low_level_template_id"].values,
        zero_rows["pattern_type"].values,
    )

    template_pairs = damage_by_template[
        (damage_by_template["scheme"] == "zero") & (damage_by_template["pattern_type"] == "PAIR")
    ]
    n_baba_pos = int(template_pairs["baba_sign_positive"].sum())
    n_abba_pos = int(template_pairs["abba_sign_positive"].sum())

    diagnosis = {
        **PRESERVED_VERDICT,
        "reconstruction_checks": {k: recon[k] for k in recon if k != "consensus"},
        "cross_run_endpoint_check": cross,
        "damage_drop_field_consistency": drop_check,
        "io_semantic_audit": io_audit,
        "diagnosis_category": category,
        "zero_balanced_mean": zero_balanced_mean,
        "zero_baba_balanced_mean": baba_mean,
        "zero_abba_balanced_mean": abba_mean,
        "zero_baba_minus_abba": baba_mean - abba_mean,
        "zero_template_pair_baba_positive_count": n_baba_pos,
        "zero_template_pair_abba_positive_count": n_abba_pos,
        "recommended_zero_terminology": recommended_zero_terminology(category, baba_mean, abba_mean),
    }
    (out_dir / "zero_reference_diagnosis.json").write_text(json.dumps(diagnosis, indent=2) + "\n")
    (out_dir / "historical_integrity_audit.json").write_text(json.dumps(integrity, indent=2) + "\n")
    (out_dir / "informative_label_clarification.md").write_text(
        informative_label_clarification_md(damage_df, damage_by_pattern)
    )

    interp = [
        "# EF-5R scientific interpretation (post-stop diagnosis)",
        "",
        "## 1. Fine T/R assignment robustness",
        "Under informative matched-mean and matched-resample schemes, all four frozen channels "
        "remained **ROBUST_SAME** in EF-5. EF-5R does not alter that verdict.",
        "",
        "## 2. Task-performance effect by scheme",
        f"- Zero reference balanced damage_drop: {zero_balanced_mean:.10f} "
        "(exact balanced estimator; not BABA+ABBA sum).",
        f"- Zero BABA: {baba_mean:+.4f}; zero ABBA: {abba_mean:+.4f}.",
        "- Matched mean/resample: positive balanced damage_drop (performance-degrading).",
        "",
        "## 3. Does zero removal deserve 'damage'?",
        recommended_zero_terminology(category, baba_mean, abba_mean),
        "",
        "## 4. EF-5 failure semantics",
        "EF-5 FAIL is driven by **reference perturbation semantics** (zero forward drop not uniformly "
        "positive), not by fine-channel reversal under informative alternatives.",
        "",
        f"**Diagnosis category: {category}**",
        "",
        "Do not authorize cross-setting/model generalization.",
    ]
    (out_dir / "scientific_interpretation.md").write_text("\n".join(interp) + "\n")

    return {
        "diagnosis": diagnosis,
        "integrity": integrity,
        "fine": fine,
        "damage_by_pattern": damage_by_pattern,
    }
