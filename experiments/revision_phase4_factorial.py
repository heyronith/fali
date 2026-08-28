"""
Phase-4A smoke: canonical 30-stratum IOI + Phase-3D P×V×C factorial machinery.

Default: N=60 (2 per stratum), seed=42. Refuse N>=900 without --allow-full.
Writes under results/revision_phase4/ only.

Does NOT reinterpret Phase-3 hypotheses from smoke template effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from revision_config import (  # noqa: E402
    PRIMARY_NAME_MOVERS,
    all_phase1_candidates,
    head_name_from_pair,
    role_for_head,
)
from revision_factorial import (  # noqa: E402
    build_factorial_fwd_hooks,
    endpoint_marginal_effects,
    factorial_interactions,
    max_abs_error,
    reconstruct_all_hybrid_z,
    reconstruct_z_head,
    three_factor_shapley,
)
from revision_interventions import build_zero_ablation_hooks  # noqa: E402
from revision_standard_ioi import (  # noqa: E402
    StandardIOIDataset,
    build_provenance,
    legacy_template_overlap_report,
    template_inventory_df,
)

Head = Tuple[int, int]

# Frozen Phase-3D hypotheses — do not redefine after seeing broad-template data.
FROZEN_HYPOTHESES = {
    "H1_L9H7": {
        "head": "L9H7",
        "primary": True,
        "predict": "phi_P ≈ 0; phi_V ≈ 0; phi_C > 0",
    },
    "H2_L10H2": {
        "head": "L10H2",
        "primary": True,
        "predict": "conditional_gain > 0; phi_P > 0; context potentially negative",
    },
    "H3_L10H10": {
        "head": "L10H10",
        "primary": True,
        "predict": "phi_P > 0; phi_C < 0; cancellation / negative gain",
    },
    "H4_L10H7": {
        "head": "L10H7",
        "primary": True,
        "predict": "phi_P large positive; conditional_effect < 0",
    },
    "secondary_L10H1": {
        "head": "L10H1",
        "primary": False,
        "predict": "suggestive context-led recruitment",
    },
}

FORBIDDEN_PREFIXES = (
    "results/raw",
    "results/summary",
    "results/revision_phase1/full/seed_",
    "results/revision_phase2/full/seed_",
    "results/revision_phase3c/full/seed_",
)

HISTORICAL_HASH_TARGETS = [
    "results/revision_phase1/full/seed_42/candidate_raw.csv",
    "results/revision_phase2/full/seed_42/phase2_combined_raw.csv",
    "results/revision_phase3c/full/seed_42/factorial_raw.csv",
    "results/raw/fali_decomposition_raw.csv",
]

Z_RECON_TOL = 1e-5
ENDPOINT_TOL_PREF = 1e-4
ENDPOINT_TOL_MAX = 1e-3
SHAPLEY_TOL = 1e-5
INTERACTION_TOL = 1e-5
SAME_LAYER_TOL = 1e-5

FUTURE_FULL_N = 900
FUTURE_SEEDS = [42, 1, 8, 22]
FUTURE_BOOTSTRAP = 10000
SOURCE_COMMIT = "e168301aa05154a14d52e5b6f98b6518200f5b1f"


def resolve_device(requested: Optional[str]) -> str:
    if requested and requested.lower() != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def file_sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_protected(repo: Path) -> Dict:
    snap = {}
    for rel in HISTORICAL_HASH_TARGETS:
        p = repo / rel
        st = p.stat() if p.exists() else None
        snap[rel] = {
            "sha256": file_sha256(p),
            "size": st.st_size if st else None,
        }
    return snap


def assert_safe_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    repo = _REPO_ROOT.resolve()
    try:
        rel = str(resolved.relative_to(repo)).replace("\\", "/")
    except ValueError:
        rel = str(resolved)
    for bad in FORBIDDEN_PREFIXES:
        if rel == bad or rel.startswith(bad):
            raise ValueError(f"Refusing protected path: {rel}")
    if "results/revision_phase4" not in rel:
        raise ValueError(f"Outputs must be under results/revision_phase4/, got {rel}")


def pkg_ver(name: str) -> Optional[str]:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return None


def compute_logit_diff(logits: torch.Tensor, dataset) -> torch.Tensor:
    batch = dataset.batch_size
    device = logits.device
    end_idx = dataset.end.to(device)
    batch_idx = torch.arange(batch, device=device)
    logits_at_end = logits[batch_idx, end_idx]
    io = logits_at_end[batch_idx, dataset.io_token_ids.to(device)]
    s = logits_at_end[batch_idx, dataset.s_token_ids.to(device)]
    return io - s


def stratum_matches_hypothesis(hname: str, row: pd.Series) -> Optional[bool]:
    """Coarse sign checks for frozen primary hypotheses (reporting only)."""
    if hname == "L9H7":
        return bool(
            abs(row["phi_pattern"]) < 0.05
            and abs(row["phi_value"]) < 0.05
            and row["phi_context"] > 0
        )
    if hname == "L10H2":
        return bool(row["conditional_gain"] > 0 and row["phi_pattern"] > 0)
    if hname == "L10H10":
        return bool(row["phi_pattern"] > 0 and row["phi_context"] < 0)
    if hname == "L10H7":
        return bool(row["phi_pattern"] > 0.5 and row["conditional_effect"] < 0)
    if hname == "L10H1":
        return bool(row["phi_context"] > 0)
    return None


def write_dataset_validation_report(dataset: StandardIOIDataset, path: Path) -> None:
    meta = dataset.metadata_df
    filt = dataset.tokenization_filter_report()
    lines = [
        "# Dataset validation report (Phase-4A smoke)",
        "",
        f"- N={dataset.num_samples} = {dataset.num_per_stratum} × 30 strata",
        f"- seed={dataset.seed}",
        f"- usable single-token names: {filt['usable_names']}",
        f"- rejected names: {len(filt['rejected_names'])}",
        f"- usable places: {filt['usable_places']} (rejected {len(filt['rejected_places'])})",
        f"- usable objects: {filt['usable_objects']} (rejected {len(filt['rejected_objects'])})",
        "",
        "## Balance",
        f"- template IDs present: {sorted(meta['low_level_template_id'].unique().tolist())}",
        f"- nunique template IDs: {meta['low_level_template_id'].nunique()} (expect 15)",
        f"- pattern types: {sorted(meta['pattern_type'].unique().tolist())}",
        f"- n strata: {meta['template_stratum'].nunique()} (expect 30)",
        f"- counts per stratum: {meta.groupby('template_stratum').size().unique().tolist()}",
        "",
        "## Token/index assertions",
        "- All StandardIOIDataset.validate() checks passed at construction.",
        "- end never on padding; S×2 / IO×1; S≠IO; decoded prompts contain names.",
        "",
    ]
    if filt["rejected_names"] or filt["rejected_places"] or filt["rejected_objects"]:
        lines.append("## Explicit tokenization rejects")
        lines.append(json.dumps(filt, indent=2))
    else:
        lines.append("## Explicit tokenization rejects")
        lines.append("- none (all canonical names/places/objects single-token under GPT-2 Small).")
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=60)
    p.add_argument("--num-per-stratum", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow-full", action="store_true")
    p.add_argument("--output-dir", default="results/revision_phase4/smoke")
    args = p.parse_args()

    # Prefer stratum balancing: N = num_per_stratum * 30
    if args.num_per_stratum > 0:
        expected_n = args.num_per_stratum * 30
        if args.num_samples != expected_n:
            # Allow explicit N that matches strata design
            if args.num_samples % 30 == 0:
                args.num_per_stratum = args.num_samples // 30
            else:
                raise SystemExit(
                    f"N={args.num_samples} must be divisible by 30 strata "
                    f"(or set --num-per-stratum)."
                )
        else:
            pass
    num_per_stratum = args.num_samples // 30

    if args.num_samples >= FUTURE_FULL_N and not args.allow_full:
        raise SystemExit(
            f"Refusing N>={FUTURE_FULL_N} without --allow-full "
            "(Phase-4A smoke/dataset validation only)."
        )

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    assert_safe_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_lines: List[str] = []

    def log(msg: str):
        print(msg, flush=True)
        log_lines.append(msg)

    pre_snap = snapshot_protected(_REPO_ROOT)
    device = resolve_device(args.device)
    if args.require_cuda and not str(device).startswith("cuda"):
        raise RuntimeError(f"--require-cuda but device={device}")

    run_config = {
        "phase": "4A",
        "num_samples": args.num_samples,
        "num_per_stratum": num_per_stratum,
        "n_strata": 30,
        "n_low_level_templates": 15,
        "seed": args.seed,
        "device": device,
        "dtype": "float32",
        "smoke": bool(args.smoke or args.num_samples <= 60),
        "experiment_name": "canonical-template / public-implementation lexical distribution",
        "lexical_inventory": {
            "source": "public Easy-Transformer implementation",
            "n_names": 99,
            "n_places": 8,
            "n_objects": 8,
            "paper_text_describes": "100 names and 20 total common place/object words",
            "templates": "15 low-level BABA match paper Appendix E",
        },
        "primary_name_movers": [head_name_from_pair(h) for h in PRIMARY_NAME_MOVERS],
        "heads": [head_name_from_pair(h) for h in all_phase1_candidates()],
        "factors": ["P=pattern", "V=value", "C=context"],
        "intervention": "full hook_z all query positions (Phase-3D machinery)",
        "frozen_hypotheses": FROZEN_HYPOTHESES,
        "future_full_n_per_seed": FUTURE_FULL_N,
        "future_seeds": FUTURE_SEEDS,
        "future_bootstrap_draws": FUTURE_BOOTSTRAP,
        "bootstrap_procedures": [
            "within_template_stratified (30 strata)",
            "template_cluster (15 templates; BABA+ABBA paired)",
        ],
        "canonical_source_commit": SOURCE_COMMIT,
        "output_dir": str(out_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "science_interpretation": "FORBIDDEN for N=60 smoke",
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    t0 = time.time()
    try:
        # Provenance + inventory + overlap (always under smoke + provenance)
        prov_dir = _REPO_ROOT / "results/revision_phase4/provenance"
        prov_dir.mkdir(parents=True, exist_ok=True)
        from dataclasses import asdict

        prov = build_provenance(
            commit_sha=SOURCE_COMMIT, retrieval_date_utc="2026-08-14"
        )
        prov_path = prov_dir / "standard_ioi_provenance.json"
        if not prov_path.exists():
            prov_path.write_text(json.dumps(asdict(prov), indent=2))

        inv = template_inventory_df()
        inv.to_csv(out_dir / "template_inventory.csv", index=False)
        overlap = legacy_template_overlap_report()
        (out_dir / "legacy_template_overlap_report.md").write_text(overlap)
        (prov_dir / "legacy_template_overlap_report.md").write_text(overlap)

        log(f"Loading gpt2-small on {device} FP32...")
        model = HookedTransformer.from_pretrained(
            "gpt2-small",
            center_writing_weights=True,
            device=device,
            dtype="float32",
        )
        model.eval()

        log(
            f"Building StandardIOIDataset N={args.num_samples} "
            f"({num_per_stratum}/stratum × 30) seed={args.seed}..."
        )
        dataset = StandardIOIDataset(
            model, num_per_stratum=num_per_stratum, seed=args.seed
        )
        assert dataset.num_samples == args.num_samples
        tokens = dataset.clean_tokens.to(device)
        meta = dataset.metadata_df.copy()
        meta.to_csv(out_dir / "dataset_metadata.csv", index=False)
        write_dataset_validation_report(dataset, out_dir / "dataset_validation_report.md")
        filt = dataset.tokenization_filter_report()
        log(f"Usable names={filt['usable_names']} rejected={len(filt['rejected_names'])}")

        candidates = all_phase1_candidates()
        layers = sorted({l for l, _ in candidates} | {l for l, _ in PRIMARY_NAME_MOVERS})
        names_filter = []
        for L in layers:
            names_filter.extend(
                [
                    f"blocks.{L}.attn.hook_pattern",
                    f"blocks.{L}.attn.hook_v",
                    f"blocks.{L}.attn.hook_z",
                ]
            )

        log("Cache A (clean)...")
        with torch.no_grad():
            logits_A, cache_A = model.run_with_cache(tokens, names_filter=names_filter)
        ios_A = compute_logit_diff(logits_A, dataset)

        log("Cache B (primary knockout)...")
        abl = build_zero_ablation_hooks(PRIMARY_NAME_MOVERS)
        with model.hooks(fwd_hooks=abl):
            with torch.no_grad():
                logits_B, cache_B = model.run_with_cache(
                    tokens, names_filter=names_filter
                )
        ios_B = compute_logit_diff(logits_B, dataset)

        raw_rows: List[Dict] = []
        recon_rows: List[Dict] = []
        same_layer_rows: List[Dict] = []

        def ios_with(hooks) -> torch.Tensor:
            with torch.no_grad():
                with model.hooks(fwd_hooks=hooks):
                    logits = model(tokens)
            return compute_logit_diff(logits, dataset)

        for layer, head in candidates:
            hname = head_name_from_pair((layer, head))
            role = role_for_head((layer, head))
            log(f"Factorial {hname}...")

            pat_A = cache_A[f"blocks.{layer}.attn.hook_pattern"]
            pat_B = cache_B[f"blocks.{layer}.attn.hook_pattern"]
            v_A = cache_A[f"blocks.{layer}.attn.hook_v"]
            v_B = cache_B[f"blocks.{layer}.attn.hook_v"]
            z_A_cached = cache_A[f"blocks.{layer}.attn.hook_z"][:, :, head, :]
            z_B_cached = cache_B[f"blocks.{layer}.attn.hook_z"][:, :, head, :]

            z_A_rec = reconstruct_z_head(pat_A, v_A, head)
            z_B_rec = reconstruct_z_head(pat_B, v_B, head)
            err_A = max_abs_error(z_A_rec, z_A_cached)
            err_B = max_abs_error(z_B_rec, z_B_cached)
            recon_rows.append(
                {
                    "head_name": hname,
                    "recon_err_A": err_A,
                    "recon_err_B": err_B,
                    "pass_A": err_A <= Z_RECON_TOL,
                    "pass_B": err_B <= Z_RECON_TOL,
                }
            )
            if err_A > Z_RECON_TOL or err_B > Z_RECON_TOL:
                log(f"  WARN recon err A={err_A:.3e} B={err_B:.3e}")

            if layer == 9:
                pdiff = max_abs_error(pat_A[:, head], pat_B[:, head])
                vdiff = max_abs_error(v_A[:, :, head], v_B[:, :, head])
                same_layer_rows.append(
                    {
                        "head_name": hname,
                        "max_pattern_A_B": pdiff,
                        "max_value_A_B": vdiff,
                        "pattern_invariant": pdiff <= SAME_LAYER_TOL,
                        "value_invariant": vdiff <= SAME_LAYER_TOL,
                    }
                )

            hybrids = reconstruct_all_hybrid_z(pat_A, pat_B, v_A, v_B, head)

            hooks_A0 = build_factorial_fwd_hooks(
                (layer, head), context_B=False, z_head=None, zero_candidate=True
            )
            hooks_B0 = build_factorial_fwd_hooks(
                (layer, head), context_B=True, z_head=None, zero_candidate=True
            )
            ios_A0 = ios_with(hooks_A0)
            ios_B0 = ios_with(hooks_B0)

            intact = ios_A - ios_A0
            conditional = ios_B - ios_B0
            gain = conditional - intact

            cell_z = {
                (0, 0): hybrids["z_00"],
                (1, 0): hybrids["z_10"],
                (0, 1): hybrids["z_01"],
                (1, 1): hybrids["z_11"],
            }
            H = {}
            for P in (0, 1):
                for V in (0, 1):
                    for C in (0, 1):
                        zh = cell_z[(P, V)]
                        hooks = build_factorial_fwd_hooks(
                            (layer, head),
                            context_B=bool(C),
                            z_head=zh,
                            zero_candidate=False,
                        )
                        ios_cell = ios_with(hooks)
                        base = ios_B0 if C else ios_A0
                        H[(P, V, C)] = ios_cell - base

            for i in range(dataset.batch_size):
                cells = {
                    f"H{P}{V}{C}": float(H[(P, V, C)][i].item())
                    for P in (0, 1)
                    for V in (0, 1)
                    for C in (0, 1)
                }
                H000 = cells["H000"]
                H111 = cells["H111"]
                intact_i = float(intact[i].item())
                cond_i = float(conditional[i].item())
                gain_i = float(gain[i].item())

                err_h000 = abs(H000 - intact_i)
                err_h111 = abs(H111 - cond_i)
                err_gain = abs((H111 - H000) - gain_i)

                phi_P, phi_V, phi_C, shap_res = three_factor_shapley(
                    cells["H000"],
                    cells["H100"],
                    cells["H010"],
                    cells["H110"],
                    cells["H001"],
                    cells["H101"],
                    cells["H011"],
                    cells["H111"],
                )
                inter = factorial_interactions(
                    cells["H000"],
                    cells["H100"],
                    cells["H010"],
                    cells["H110"],
                    cells["H001"],
                    cells["H101"],
                    cells["H011"],
                    cells["H111"],
                )
                marg = endpoint_marginal_effects(
                    cells["H000"],
                    cells["H100"],
                    cells["H010"],
                    cells["H110"],
                    cells["H001"],
                    cells["H101"],
                    cells["H011"],
                    cells["H111"],
                )

                mrow = meta.iloc[i]
                raw_rows.append(
                    {
                        "seed": args.seed,
                        "example_idx": int(mrow["example_idx"]),
                        "low_level_template_id": int(mrow["low_level_template_id"]),
                        "pattern_type": str(mrow["pattern_type"]),
                        "template_stratum": str(mrow["template_stratum"]),
                        "head_name": hname,
                        "layer": layer,
                        "head": head,
                        "head_role": role,
                        "intact_effect": intact_i,
                        "conditional_effect": cond_i,
                        "conditional_gain": gain_i,
                        **cells,
                        "phi_pattern": phi_P,
                        "phi_value": phi_V,
                        "phi_context": phi_C,
                        "shapley_residual": shap_res,
                        **inter,
                        **marg,
                        "err_H000_vs_intact": err_h000,
                        "err_H111_vs_conditional": err_h111,
                        "err_gain_identity": err_gain,
                        "recon_err_A": err_A,
                        "recon_err_B": err_B,
                    }
                )

        raw_df = pd.DataFrame(raw_rows)
        raw_df.to_csv(out_dir / "factorial_raw.csv", index=False)

        def frac_pos(s):
            return float((s > 0).mean())

        sum_rows = []
        for hname, sub in raw_df.groupby("head_name", sort=False):
            role = sub["head_role"].iloc[0]
            sum_rows.append(
                {
                    "head_name": hname,
                    "head_role": role,
                    "mean_intact_effect": float(sub["intact_effect"].mean()),
                    "mean_conditional_effect": float(sub["conditional_effect"].mean()),
                    "mean_conditional_gain": float(sub["conditional_gain"].mean()),
                    "mean_phi_pattern": float(sub["phi_pattern"].mean()),
                    "mean_phi_value": float(sub["phi_value"].mean()),
                    "mean_phi_context": float(sub["phi_context"].mean()),
                    "fraction_pattern_positive": frac_pos(sub["phi_pattern"]),
                    "fraction_value_positive": frac_pos(sub["phi_value"]),
                    "fraction_context_positive": frac_pos(sub["phi_context"]),
                    "max_endpoint_error": float(
                        max(
                            sub["err_H000_vs_intact"].max(),
                            sub["err_H111_vs_conditional"].max(),
                            sub["err_gain_identity"].max(),
                        )
                    ),
                    "max_shapley_residual": float(sub["shapley_residual"].abs().max()),
                    "max_interaction_residual": float(
                        sub["interaction_recon_residual"].abs().max()
                    ),
                }
            )
        sum_df = pd.DataFrame(sum_rows)
        sum_df.to_csv(out_dir / "factorial_summary.csv", index=False)

        # Per head × template × pattern (do not average before checking signs)
        tmpl_rows = []
        for keys, sub in raw_df.groupby(
            ["head_name", "low_level_template_id", "pattern_type"], sort=True
        ):
            hname, tid, ptype = keys
            row = {
                "head_name": hname,
                "low_level_template_id": int(tid),
                "pattern_type": ptype,
                "template_stratum": f"T{int(tid):02d}_{ptype}",
                "n": len(sub),
                "conditional_effect": float(sub["conditional_effect"].mean()),
                "conditional_gain": float(sub["conditional_gain"].mean()),
                "phi_pattern": float(sub["phi_pattern"].mean()),
                "phi_value": float(sub["phi_value"].mean()),
                "phi_context": float(sub["phi_context"].mean()),
            }
            tmpl_rows.append(row)
        tmpl_df = pd.DataFrame(tmpl_rows)
        tmpl_df.to_csv(out_dir / "template_factorial_smoke_summary.csv", index=False)

        # Fraction of 30 strata matching each frozen hypothesis (reporting only)
        hyp_lines = [
            "# Frozen hypothesis stratum match fractions (N=60 smoke — NOT for inference)",
            "",
            "Phase-3D hypotheses remain frozen. Counts below are execution telemetry only.",
            "",
        ]
        for hyp_name, hyp in FROZEN_HYPOTHESES.items():
            hname = hyp["head"]
            sub = tmpl_df[tmpl_df["head_name"] == hname]
            assert len(sub) == 30, f"{hname} expected 30 strata, got {len(sub)}"
            matches = []
            for _, r in sub.iterrows():
                m = stratum_matches_hypothesis(hname, r)
                if m is not None:
                    matches.append(bool(m))
            frac = float(np.mean(matches)) if matches else float("nan")
            hyp_lines.append(
                f"- {hyp_name} ({hname}): {frac:.3f} of {len(matches)} strata "
                f"(predict: {hyp['predict']})"
            )
        (out_dir / "frozen_hypothesis_stratum_fractions.md").write_text(
            "\n".join(hyp_lines) + "\n"
        )

        recon_df = pd.DataFrame(recon_rows)
        same_df = pd.DataFrame(same_layer_rows)

        max_recon = float(
            max(recon_df["recon_err_A"].max(), recon_df["recon_err_B"].max())
        )
        max_h000 = float(raw_df["err_H000_vs_intact"].max())
        max_h111 = float(raw_df["err_H111_vs_conditional"].max())
        max_gain_err = float(raw_df["err_gain_identity"].max())
        max_shap = float(raw_df["shapley_residual"].abs().max())
        max_inter = float(raw_df["interaction_recon_residual"].abs().max())

        (out_dir / "endpoint_identity_report.md").write_text(
            "# Endpoint identity report\n\n"
            f"- preferred tol: {ENDPOINT_TOL_PREF}, max allowed: {ENDPOINT_TOL_MAX}\n"
            f"- max |H000 − intact|: {max_h000:.3e}\n"
            f"- max |H111 − conditional|: {max_h111:.3e}\n"
            f"- max |(H111−H000) − gain|: {max_gain_err:.3e}\n\n"
            + sum_df[
                [
                    "head_name",
                    "mean_intact_effect",
                    "mean_conditional_effect",
                    "max_endpoint_error",
                ]
            ].to_string(index=False)
            + "\n"
        )
        (out_dir / "same_layer_report.md").write_text(
            "# Same-layer (Layer 9) P/V invariants\n\n"
            f"- tol: {SAME_LAYER_TOL}\n\n"
            + (same_df.to_string(index=False) if len(same_df) else "no layer-9 heads")
            + "\n"
        )

        finite_cols = list(raw_df.select_dtypes(include=[np.number]).columns)
        arr = raw_df[finite_cols].to_numpy(dtype=np.float64)
        has_nan = bool(np.isnan(arr).any() or np.isinf(arr).any())

        endpoint_ok = max(max_h000, max_h111, max_gain_err) <= ENDPOINT_TOL_MAX
        endpoint_pref = max(max_h000, max_h111, max_gain_err) <= ENDPOINT_TOL_PREF
        shap_ok = max_shap <= SHAPLEY_TOL
        inter_ok = max_inter <= INTERACTION_TOL
        recon_ok = max_recon <= Z_RECON_TOL
        l9 = same_df[same_df["head_name"].isin(["L9H7", "L9H0"])] if len(same_df) else same_df
        l9_ok = True
        if len(l9):
            l9_ok = bool(
                l9["pattern_invariant"].all() and l9["value_invariant"].all()
            )

        strata_ok = bool(
            meta["low_level_template_id"].nunique() == 15
            and meta["template_stratum"].nunique() == 30
            and set(meta["pattern_type"]) == {"BABA", "ABBA"}
            and (meta.groupby("template_stratum").size() == num_per_stratum).all()
        )

        post_snap = snapshot_protected(_REPO_ROOT)
        hist_ok = pre_snap == post_snap

        checks = {
            "no_nan_inf": not has_nan,
            "z_reconstruction_ok": recon_ok,
            "endpoint_identity_ok": endpoint_ok,
            "endpoint_preferred_tol": endpoint_pref,
            "shapley_residual_ok": shap_ok,
            "interaction_identity_ok": inter_ok,
            "all_12_heads": raw_df["head_name"].nunique() == 12,
            "n_rows": len(raw_df) == args.num_samples * 12,
            "all_30_strata": strata_ok,
            "l9_pv_invariant": l9_ok,
            "historical_untouched": hist_ok,
            "cuda_ok": (not args.require_cuda) or str(device).startswith("cuda"),
        }

        report = [
            "# Phase-4A validation report",
            "",
            f"- device: {device}",
            f"- N={args.num_samples} seed={args.seed} ({num_per_stratum}×30 strata)",
            f"- max z recon err: {max_recon:.3e}",
            f"- max H000/intact err: {max_h000:.3e}",
            f"- max H111/conditional err: {max_h111:.3e}",
            f"- max gain identity err: {max_gain_err:.3e}",
            f"- max Shapley residual: {max_shap:.3e}",
            f"- max interaction recon residual: {max_inter:.3e}",
            "",
            "## Checks",
        ]
        for k, v in checks.items():
            report.append(f"- {'PASS' if v else 'FAIL'}: `{k}`")
        report.append("\n## Factorial summary (smoke — not for scientific inference)\n")
        report.append(sum_df.to_string(index=False))
        (out_dir / "validation_report.md").write_text("\n".join(report) + "\n")

        env = {
            "python": sys.version,
            "platform": platform.platform(),
            "device": device,
            "cuda_device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "torch": torch.__version__,
            "transformer_lens": pkg_ver("transformer-lens"),
            "transformers": pkg_ver("transformers"),
            "checks": checks,
            "tokenization_filter": filt,
            "max_z_recon_err": max_recon,
            "max_H000_err": max_h000,
            "max_H111_err": max_h111,
            "max_gain_err": max_gain_err,
            "max_shapley_residual": max_shap,
            "max_interaction_residual": max_inter,
            "endpoint_preferred_tol_pass": endpoint_pref,
            "protected_pre": pre_snap,
            "protected_post": post_snap,
            "elapsed_sec": time.time() - t0,
        }
        (out_dir / "environment.json").write_text(
            json.dumps(env, indent=2, default=str)
        )

        if not endpoint_ok or not shap_ok or not recon_ok or not inter_ok:
            log("ACCEPTANCE FAILED — core identity failure; STOP")
            for k, v in checks.items():
                if not v:
                    log(f"  FAIL: {k}")
            raise SystemExit(2)

        if not all(checks.values()):
            log("ACCEPTANCE FAILED")
            for k, v in checks.items():
                if not v:
                    log(f"  FAIL: {k}")
            raise SystemExit(2)

        log("ACCEPTANCE PASSED")
        if not endpoint_pref:
            log(
                f"NOTE: endpoint errors within max tol {ENDPOINT_TOL_MAX} "
                f"but above preferred {ENDPOINT_TOL_PREF}"
            )
        log(f"Done in {time.time()-t0:.1f}s -> {out_dir}")

    except Exception:
        log(traceback.format_exc())
        raise
    finally:
        (out_dir / "run_log.txt").write_text("\n".join(log_lines) + "\n")


if __name__ == "__main__":
    main()
