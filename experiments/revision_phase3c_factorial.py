"""
Phase-3C smoke/full: full-head P×V×C factorial Shapley of conditional_gain.

Default: smoke N=8. Refuse N>=200 without --allow-full.
Writes under results/revision_phase3c/ only.
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
from revision_dataset import (  # noqa: E402
    build_dataset_metadata,
    make_validated_ioi_dataset,
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

Head = Tuple[int, int]

REPORTING_PANEL = ["L10H2", "L10H1", "L10H10", "L9H7", "L10H7", "L10H6"]

FORBIDDEN_PREFIXES = (
    "results/raw",
    "results/summary",
    "results/revision_phase1/full/seed_",
    "results/revision_phase2/full/seed_",
)

HISTORICAL_HASH_TARGETS = [
    "results/revision_phase1/full/seed_42/candidate_raw.csv",
    "results/revision_phase2/full/seed_42/phase2_combined_raw.csv",
    "results/raw/fali_decomposition_raw.csv",
]

Z_RECON_TOL = 1e-5
ENDPOINT_TOL_PREF = 1e-4
ENDPOINT_TOL_MAX = 1e-3
SHAPLEY_TOL = 1e-5
INTERACTION_TOL = 1e-5
SAME_LAYER_TOL = 1e-5



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
    if "results/revision_phase3c" not in rel:
        raise ValueError(f"Outputs must be under results/revision_phase3c/, got {rel}")


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


def run_ios(model, tokens, hooks=None) -> torch.Tensor:
    with torch.no_grad():
        if hooks:
            with model.hooks(fwd_hooks=hooks):
                logits = model(tokens)
        else:
            logits = model(tokens)
    return compute_logit_diff(logits, model._phase3c_dataset)  # set below


def align_dataset(meta: pd.DataFrame, phase1_meta: Path, report_path: Path) -> None:
    lines = ["# Dataset alignment", ""]
    if not phase1_meta.exists():
        raise FileNotFoundError(phase1_meta)
    ref = pd.read_csv(phase1_meta).head(len(meta))
    cols = ["example_idx", "prompt", "S", "IO", "template_id", "end_position"]
    mismatches = []
    for c in cols:
        for i, (a, b) in enumerate(
            zip(meta[c].astype(str), ref[c].astype(str))
        ):
            if a != b:
                mismatches.append((i, c, a, b))
    if mismatches:
        lines.append(f"FAIL: {len(mismatches)} mismatches")
        report_path.write_text("\n".join(lines))
        raise AssertionError(f"Dataset mismatch ({len(mismatches)})")
    lines.append(f"PASS vs `{phase1_meta}` ({len(meta)} examples)")
    report_path.write_text("\n".join(lines) + "\n")


def load_phase2_end_query_causal(seed: int, n: int) -> pd.DataFrame:
    """
    Per-example end-query causal_total from Phase-2 full seed (first n examples).
    """
    path = (
        _REPO_ROOT
        / f"results/revision_phase2/full/seed_{seed}/phase2_combined_raw.csv"
    )
    if not path.exists():
        # fall back to smoke summary means only
        smoke = _REPO_ROOT / "results/revision_phase2/smoke/causal_score_hybrid_summary.csv"
        if smoke.exists():
            s = pd.read_csv(smoke)
            s = s.rename(columns={"causal_total": "existing_end_query_causal_total"})
            # smoke may lack causal_total column name - compute from F11-F00 if needed
            if "existing_end_query_causal_total" not in s.columns:
                if "F11" in s.columns and "F00" in s.columns:
                    s["existing_end_query_causal_total"] = s["F11"] - s["F00"]
                elif "causal_target" in s.columns:
                    s["existing_end_query_causal_total"] = (
                        s["causal_target"] + s["causal_competing"]
                    )
            return s[["head_name", "existing_end_query_causal_total"]]
        return pd.DataFrame(columns=["head_name", "example_idx", "existing_end_query_causal_total"])

    df = pd.read_csv(path)
    df = df[df["example_idx"] < n][
        ["example_idx", "head_name", "causal_total", "q_clean_effect", "k_clean_effect"]
    ].copy()
    df = df.rename(columns={"causal_total": "existing_end_query_causal_total"})
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--allow-full", action="store_true")
    p.add_argument("--output-dir", default="results/revision_phase3c/smoke")
    p.add_argument("--phase1-meta", default="")
    args = p.parse_args()

    if args.num_samples >= 200 and not args.allow_full:
        raise SystemExit("Refusing N>=200 without --allow-full (Phase-3C smoke only).")

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
        "phase": "3C",
        "num_samples": args.num_samples,
        "seed": args.seed,
        "device": device,
        "dtype": "float32",
        "smoke": bool(args.smoke or args.num_samples <= 8),
        "primary_name_movers": [head_name_from_pair(h) for h in PRIMARY_NAME_MOVERS],
        "heads": [head_name_from_pair(h) for h in all_phase1_candidates()],
        "reporting_panel": REPORTING_PANEL,
        "factors": ["P=pattern", "V=value", "C=context"],
        "intervention": "full hook_z all query positions",
        "output_dir": str(out_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    t0 = time.time()
    try:
        log(f"Loading gpt2-small on {device} FP32...")
        model = HookedTransformer.from_pretrained(
            "gpt2-small",
            center_writing_weights=True,
            device=device,
            dtype="float32",
        )
        model.eval()

        dataset = make_validated_ioi_dataset(
            model, num_samples=args.num_samples, seed=args.seed, template_style="mixed"
        )
        model._phase3c_dataset = dataset  # for run_ios helper
        tokens = dataset.clean_tokens.to(device)
        meta = build_dataset_metadata(dataset)
        meta.to_csv(out_dir / "dataset_metadata.csv", index=False)

        if args.phase1_meta:
            phase1_meta = Path(args.phase1_meta)
        elif args.num_samples <= 8:
            phase1_meta = _REPO_ROOT / "results/revision_phase1/smoke/dataset_metadata.csv"
        else:
            phase1_meta = (
                _REPO_ROOT
                / f"results/revision_phase1/full/seed_{args.seed}/dataset_metadata.csv"
            )
        align_dataset(meta, phase1_meta, out_dir / "dataset_alignment_report.md")
        log(f"Aligned to {phase1_meta}")
        if args.num_samples == 200:
            n_a = int((meta["template_id"] == "A").sum())
            n_b = int((meta["template_id"] == "B").sum())
            assert n_a == 100 and n_b == 100, f"Expected 100/100 A/B, got {n_a}/{n_b}"
            log("Template counts: A=100 B=100")

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

        # Cache A
        log("Cache A (clean)...")
        with torch.no_grad():
            logits_A, cache_A = model.run_with_cache(tokens, names_filter=names_filter)
        ios_A = compute_logit_diff(logits_A, dataset)

        # Cache B
        log("Cache B (primary knockout)...")
        abl = build_zero_ablation_hooks(PRIMARY_NAME_MOVERS)
        with model.hooks(fwd_hooks=abl):
            with torch.no_grad():
                logits_B, cache_B = model.run_with_cache(
                    tokens, names_filter=names_filter
                )
        ios_B = compute_logit_diff(logits_B, dataset)

        phase2_merge = load_phase2_end_query_causal(args.seed, args.num_samples)

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

            # Same-layer invariants
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

            # Context A zero / B zero baselines for this head
            hooks_A0 = build_factorial_fwd_hooks(
                (layer, head), context_B=False, z_head=None, zero_candidate=True
            )
            hooks_B0 = build_factorial_fwd_hooks(
                (layer, head), context_B=True, z_head=None, zero_candidate=True
            )
            ios_A0 = ios_with(hooks_A0)
            ios_B0 = ios_with(hooks_B0)

            # Intact / conditional from same-run endpoints
            intact = ios_A - ios_A0
            conditional = ios_B - ios_B0
            gain = conditional - intact

            # Eight cells: (P,V,C) with z injections
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

            # Also verify H000 using reconstructed z_00 vs natural A
            # (identity checks vs intact/conditional)

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

                # Phase-2 merge
                end_q = float("nan")
                q_eff = float("nan")
                k_eff = float("nan")
                if "example_idx" in phase2_merge.columns:
                    m = phase2_merge[
                        (phase2_merge["example_idx"] == i)
                        & (phase2_merge["head_name"] == hname)
                    ]
                    if len(m):
                        end_q = float(m["existing_end_query_causal_total"].iloc[0])
                        if "q_clean_effect" in m.columns:
                            q_eff = float(m["q_clean_effect"].iloc[0])
                            k_eff = float(m["k_clean_effect"].iloc[0])

                pattern_B_VB = marg["pattern_effect_B_VB"]
                raw_rows.append(
                    {
                        "seed": args.seed,
                        "example_idx": i,
                        "template_id": meta.loc[i, "template_id"],
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
                        "existing_end_query_causal_total": end_q,
                        "existing_q_clean_effect": q_eff,
                        "existing_k_clean_effect": k_eff,
                        "phi_pattern_minus_end_query_causal": phi_P - end_q
                        if np.isfinite(end_q)
                        else float("nan"),
                        "pattern_effect_B_VB_minus_end_query": pattern_B_VB - end_q
                        if np.isfinite(end_q)
                        else float("nan"),
                    }
                )

        raw_df = pd.DataFrame(raw_rows)
        raw_df.to_csv(out_dir / "factorial_raw.csv", index=False)

        # Summaries
        def frac_pos(s):
            return float((s > 0).mean())

        sum_rows = []
        for hname, sub in raw_df.groupby("head_name", sort=False):
            role = sub["head_role"].iloc[0]
            end_q_mean = float(sub["existing_end_query_causal_total"].mean())
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
                    "mean_main_P": float(sub["main_P"].mean()),
                    "mean_main_V": float(sub["main_V"].mean()),
                    "mean_main_C": float(sub["main_C"].mean()),
                    "mean_interaction_PV": float(sub["interaction_PV"].mean()),
                    "mean_interaction_PC": float(sub["interaction_PC"].mean()),
                    "mean_interaction_VC": float(sub["interaction_VC"].mean()),
                    "mean_interaction_PVC": float(sub["interaction_PVC"].mean()),
                    "mean_pattern_effect_B_VB": float(sub["pattern_effect_B_VB"].mean()),
                    "mean_value_effect_B_PB": float(sub["value_effect_B_PB"].mean()),
                    "mean_context_effect_PB_VB": float(sub["context_effect_PB_VB"].mean()),
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
                    "existing_end_query_causal_total": end_q_mean,
                    "mean_phi_pattern_minus_end_query": float(
                        sub["phi_pattern_minus_end_query_causal"].mean()
                    ),
                    "mean_pattern_B_VB_minus_end_query": float(
                        sub["pattern_effect_B_VB_minus_end_query"].mean()
                    ),
                }
            )
        sum_df = pd.DataFrame(sum_rows)
        sum_df.to_csv(out_dir / "factorial_summary.csv", index=False)

        recon_df = pd.DataFrame(recon_rows)
        same_df = pd.DataFrame(same_layer_rows)

        # Reports
        max_recon = float(
            max(recon_df["recon_err_A"].max(), recon_df["recon_err_B"].max())
        )
        max_h000 = float(raw_df["err_H000_vs_intact"].max())
        max_h111 = float(raw_df["err_H111_vs_conditional"].max())
        max_gain_err = float(raw_df["err_gain_identity"].max())
        max_shap = float(raw_df["shapley_residual"].abs().max())
        max_inter = float(raw_df["interaction_recon_residual"].abs().max())

        (out_dir / "z_reconstruction_report.md").write_text(
            "# z reconstruction report\n\n"
            f"- tolerance: {Z_RECON_TOL}\n"
            f"- max err A/B: {max_recon:.3e}\n\n"
            + recon_df.to_string(index=False)
            + "\n"
        )
        (out_dir / "endpoint_identity_report.md").write_text(
            "# Endpoint identity report\n\n"
            f"- preferred tol: {ENDPOINT_TOL_PREF}, max allowed: {ENDPOINT_TOL_MAX}\n"
            f"- max |H000 − intact|: {max_h000:.3e}\n"
            f"- max |H111 − conditional|: {max_h111:.3e}\n"
            f"- max |(H111−H000) − gain|: {max_gain_err:.3e}\n\n"
            + sum_df[
                ["head_name", "mean_intact_effect", "mean_conditional_effect", "max_endpoint_error"]
            ].to_string(index=False)
            + "\n"
        )
        (out_dir / "shapley_validation_report.md").write_text(
            "# Shapley + interaction identity validation\n\n"
            f"- Shapley tol: {SHAPLEY_TOL}; max |residual|: {max_shap:.3e}\n"
            f"- Interaction recon tol: {INTERACTION_TOL}; max |residual|: {max_inter:.3e}\n\n"
            + sum_df[
                [
                    "head_name",
                    "mean_phi_pattern",
                    "mean_phi_value",
                    "mean_phi_context",
                    "mean_conditional_gain",
                    "max_shapley_residual",
                    "max_interaction_residual",
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

        # Phase-2 merge summary (primary: pattern_effect_B_VB vs end-query)
        merge_sum = sum_df[
            [
                "head_name",
                "mean_pattern_effect_B_VB",
                "existing_end_query_causal_total",
                "mean_pattern_B_VB_minus_end_query",
                "mean_phi_pattern",
                "mean_phi_pattern_minus_end_query",
                "mean_conditional_gain",
            ]
        ].copy()
        merge_sum.to_csv(out_dir / "phase2_merge_summary.csv", index=False)

        def head_report(name: str, path: Path, extra: str = ""):
            sub = sum_df[sum_df["head_name"] == name]
            if not len(sub):
                path.write_text(f"# {name}\n\nmissing\n")
                return
            r = sub.iloc[0]
            path.write_text(
                f"# {name} smoke report (N={args.num_samples} — not for inference)\n\n"
                f"- intact: {r['mean_intact_effect']:.4f}\n"
                f"- conditional: {r['mean_conditional_effect']:.4f}\n"
                f"- gain: {r['mean_conditional_gain']:.4f}\n"
                f"- phi_pattern: {r['mean_phi_pattern']:.4f}\n"
                f"- phi_value: {r['mean_phi_value']:.4f}\n"
                f"- phi_context: {r['mean_phi_context']:.4f}\n"
                f"- end-query causal_total (Phase-2): {r['existing_end_query_causal_total']:.4f}\n"
                f"- phi_pattern − end_query: {r['mean_phi_pattern_minus_end_query']:.4f}\n"
                f"{extra}\n"
            )

        head_report(
            "L10H2",
            out_dir / "L10H2_smoke_report.md",
            "Question: pattern vs value vs context for +gain?",
        )
        head_report(
            "L10H10",
            out_dir / "L10H10_smoke_report.md",
            "Question: positive pattern outweighed by negative context/value?",
        )
        head_report(
            "L9H7",
            out_dir / "L9H7_smoke_report.md",
            "Prediction: phi_pattern≈0, phi_value≈0, gain≈phi_context.",
        )
        head_report(
            "L10H7",
            out_dir / "L10H7_smoke_report.md",
            "Question: which factor explains large +gain with negative conditional_effect?",
        )

        # Acceptance
        finite_cols = [
            c
            for c in raw_df.select_dtypes(include=[np.number]).columns
            if "existing_" not in c and "minus_end" not in c
        ]
        arr = raw_df[finite_cols].to_numpy(dtype=np.float64)
        has_nan = bool(np.isnan(arr).any() or np.isinf(arr).any())

        endpoint_ok = max(max_h000, max_h111, max_gain_err) <= ENDPOINT_TOL_MAX
        endpoint_pref = max(max_h000, max_h111, max_gain_err) <= ENDPOINT_TOL_PREF
        shap_ok = max_shap <= SHAPLEY_TOL
        inter_ok = max_inter <= INTERACTION_TOL
        recon_ok = max_recon <= Z_RECON_TOL
        l9 = same_df[same_df["head_name"] == "L9H7"] if len(same_df) else same_df
        l9_ok = True
        if len(l9):
            l9_ok = bool(l9.iloc[0]["pattern_invariant"] and l9.iloc[0]["value_invariant"])

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
            "l9h7_pv_invariant": l9_ok,
            "historical_untouched": hist_ok,
            "cuda_ok": (not args.require_cuda) or str(device).startswith("cuda"),
        }

        report = [
            "# Phase-3C validation report",
            "",
            f"- device: {device}",
            f"- N={args.num_samples} seed={args.seed}",
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
        report.append("\n## Factorial summary\n")
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
            "max_z_recon_err": max_recon,
            "max_H000_err": max_h000,
            "max_H111_err": max_h111,
            "max_shapley_residual": max_shap,
            "max_interaction_residual": max_inter,
            "endpoint_preferred_tol_pass": endpoint_pref,
            "protected_pre": pre_snap,
            "protected_post": post_snap,
            "elapsed_sec": time.time() - t0,
        }
        (out_dir / "environment.json").write_text(json.dumps(env, indent=2, default=str))

        # Hard stop on core identities
        if not endpoint_ok or not shap_ok or not recon_ok or not inter_ok:
            log("ACCEPTANCE FAILED — endpoint/Shapley/recon/interaction core failure; STOP")
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
