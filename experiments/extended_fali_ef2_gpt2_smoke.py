#!/usr/bin/env python3
"""
EF-2 — GPT-2 small IOI Extended-FALI implementation smoke (N=8, seed=42).

Validation only. Not scientific evidence. Writes under results/extended_fali/ef2_gpt2_smoke/.
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

import pandas as pd
import torch
from transformer_lens import HookedTransformer

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from extended_fali_gpt2 import (  # noqa: E402
    ENDPOINT_TOL,
    HISTORICAL_H_COLS,
    HISTORICAL_TOL,
    INJECTION_SCOPE_HEADS,
    INJECTION_SCOPE_TOL,
    LEAK_TOL,
    OWEN_TOL,
    POST_END_TOL,
    QUOTIENT_TOL,
    SAME_LAYER_HEADS,
    SAME_LAYER_OWEN_TOL,
    SAME_LAYER_SCORE_TOL,
    SCORE_PATTERN_TOL,
    Z_RECON_TOL,
    align_dataset_to_reference,
    build_ioi_partition,
    cache_names_filter,
    compare_historical_cells,
    compare_historical_shapley,
    compute_ios,
    descriptive_fali_diag,
    extract_head_cache,
    historical_quotient_cells,
    max_valid_coord_error,
    owen_diagnostics,
    owen_from_cell_table,
    partition_diagnostics,
    run_injection_scope_check,
    run_live_leak_check,
    run_post_end_causality_check,
    validate_score_pattern,
    validate_z_reconstruction,
)
from extended_fali_interventions import (  # noqa: E402
    CellTable,
    enumerate_extended_fali_cells,
    score_pattern_z_pipeline,
)
from revision_config import (  # noqa: E402
    PRIMARY_NAME_MOVERS,
    all_phase1_candidates,
    head_name_from_pair,
    role_for_head,
)
from revision_dataset import build_dataset_metadata, make_validated_ioi_dataset  # noqa: E402
from revision_factorial import build_factorial_fwd_hooks  # noqa: E402
from revision_interventions import build_zero_ablation_hooks  # noqa: E402

Head = Tuple[int, int]

FORBIDDEN_PREFIXES = (
    "results/raw",
    "results/summary",
    "results/revision_phase1/full/seed_",
    "results/revision_phase2/full/seed_",
    "results/revision_phase3c",
)

HISTORICAL_HASH_TARGETS = [
    "results/revision_phase1/full/seed_42/candidate_raw.csv",
    "results/revision_phase2/full/seed_42/phase2_combined_raw.csv",
    "results/revision_phase3c/smoke/factorial_raw.csv",
    "results/raw/fali_decomposition_raw.csv",
]

PHASE3C_SMOKE_META = _REPO_ROOT / "results/revision_phase3c/smoke/dataset_metadata.csv"
PHASE3C_SMOKE_FACTORIAL = _REPO_ROOT / "results/revision_phase3c/smoke/factorial_raw.csv"

N_EXPECTED = 8
SEED = 42
N_HEADS = 12
N_CELLS = 32


def resolve_device(requested: Optional[str]) -> str:
    if requested and requested.lower() != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


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
        snap[rel] = {"sha256": file_sha256(p), "size": st.st_size if st else None}
    return snap


def assert_safe_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    repo = _REPO_ROOT.resolve()
    try:
        rel = str(resolved.relative_to(repo)).replace("\\", "/")
    except ValueError:
        rel = str(resolved)
    for bad in FORBIDDEN_PREFIXES:
        if rel == bad or rel.startswith(bad + "/"):
            raise ValueError(f"Refusing protected path: {rel}")
    if "results/extended_fali/ef2_gpt2_smoke" not in rel:
        raise ValueError(f"Outputs must be under results/extended_fali/ef2_gpt2_smoke/, got {rel}")


def pkg_ver(name: str) -> Optional[str]:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return None


def ios_with(model, tokens, dataset, hooks) -> torch.Tensor:
    with torch.no_grad():
        with model.hooks(fwd_hooks=hooks):
            logits = model(tokens)
    return compute_ios(logits, dataset)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=N_EXPECTED)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default="auto")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--output-dir", default="results/extended_fali/ef2_gpt2_smoke")
    p.add_argument("--reference-meta", default=str(PHASE3C_SMOKE_META))
    p.add_argument("--historical-factorial", default=str(PHASE3C_SMOKE_FACTORIAL))
    args = p.parse_args()

    if args.num_samples != N_EXPECTED or args.seed != SEED:
        raise SystemExit(f"EF-2 smoke requires N={N_EXPECTED} seed={SEED}")

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    assert_safe_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_snap = snapshot_protected(_REPO_ROOT)
    device = resolve_device(args.device)
    if args.require_cuda and not str(device).startswith("cuda"):
        raise RuntimeError(f"--require-cuda but device={device}")

    validation: Dict = {
        "criteria": {},
        "overall_pass": False,
    }
    t0 = time.time()

    try:
        print(f"Loading gpt2-small on {device} FP32...", flush=True)
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
        tokens = dataset.clean_tokens.to(device)
        meta = build_dataset_metadata(dataset)
        meta.to_csv(out_dir / "dataset_metadata.csv", index=False)

        ref_meta = pd.read_csv(args.reference_meta)
        align_status, align_lines = align_dataset_to_reference(
            meta, ref_meta, token_tensor=dataset.clean_tokens
        )
        (out_dir / "dataset_alignment_report.md").write_text(
            "# Dataset alignment (EF-2)\n\n" + "\n".join(align_lines) + "\n"
        )
        validation["dataset_alignment"] = align_status
        if align_status != "PASS":
            raise AssertionError("Dataset alignment FAILED")

        partition = build_ioi_partition(dataset, device)
        part_diag = partition_diagnostics(partition)
        validation["batch_partition"] = {
            "final_queries": [int(x) for x in dataset.end.tolist()],
            "target_keys": [int(x) for x in dataset.IO.tolist()],
            "valid_lengths": [int(x) for x in (dataset.end + 1).tolist()],
            **part_diag,
        }
        if not (
            all(e == 0 for e in part_diag["coverage_error_per_example"])
            and all(e == 0 for e in part_diag["overlap_count_per_example"])
            and all(e == 1 for e in part_diag["target_count_per_example"])
        ):
            raise AssertionError("Partition diagnostics failed")

        candidates = all_phase1_candidates()
        layers = sorted({l for l, _ in candidates} | {l for l, _ in PRIMARY_NAME_MOVERS})
        names_filter = cache_names_filter(layers)
        primary_hooks = build_zero_ablation_hooks(PRIMARY_NAME_MOVERS)

        print("Cache A (clean)...", flush=True)
        with torch.no_grad():
            _, cache_A = model.run_with_cache(tokens, names_filter=names_filter)
        print("Cache B (primary knockout)...", flush=True)
        with model.hooks(fwd_hooks=primary_hooks):
            with torch.no_grad():
                _, cache_B = model.run_with_cache(tokens, names_filter=names_filter)

        hist_df = pd.read_csv(args.historical_factorial)
        cell_rows: List[Dict] = []
        attr_rows: List[Dict] = []
        bridge_rows: List[Dict] = []
        head_summaries: List[Dict] = []

        max_score_pat_err = 0.0
        max_z_err = 0.0
        max_endpoint_err = 0.0
        max_owen_res = 0.0
        max_quotient_err = 0.0
        max_hist_h = 0.0
        max_hist_phi = 0.0
        l9_score_v_max: Dict[str, float] = {}
        l9_owen_max: Dict[str, float] = {}
        injection_scope: Dict[str, Dict] = {}
        live_leak: Dict[str, Dict] = {}
        post_end_err = 0.0

        for layer, head in candidates:
            hname = head_name_from_pair((layer, head))
            role = role_for_head((layer, head))
            print(f"Candidate {hname}...", flush=True)

            scores_a, pat_a, v_a, z_a = extract_head_cache(cache_A, layer, head)
            scores_b, pat_b, v_b, z_b = extract_head_cache(cache_B, layer, head)

            err_pa, mask_pa = validate_score_pattern(scores_a, pat_a, partition)
            err_pb, mask_pb = validate_score_pattern(scores_b, pat_b, partition)
            max_score_pat_err = max(max_score_pat_err, err_pa, err_pb, mask_pa, mask_pb)

            p_a = pat_a  # use reconstructed for z check
            from extended_fali_interventions import masked_softmax

            p_a_rec = masked_softmax(
                scores_a.masked_fill(~partition.valid_mask, float("-inf")), partition.valid_mask
            )
            p_b_rec = masked_softmax(
                scores_b.masked_fill(~partition.valid_mask, float("-inf")), partition.valid_mask
            )
            err_za = validate_z_reconstruction(p_a_rec, v_a, z_a, dataset.end)
            err_zb = validate_z_reconstruction(p_b_rec, v_b, z_b, dataset.end)
            max_z_err = max(max_z_err, err_za, err_zb)

            if layer == 9 and hname in SAME_LAYER_HEADS:
                sv = max_valid_coord_error(scores_a, scores_b, partition.valid_mask)
                vv = float((v_a - v_b).abs().max().item())
                l9_score_v_max[hname] = max(sv, vv)

            hooks_a0 = build_factorial_fwd_hooks(
                (layer, head), context_B=False, z_head=None, zero_candidate=True
            )
            hooks_b0 = build_factorial_fwd_hooks(
                (layer, head), context_B=True, z_head=None, zero_candidate=True
            )
            hooks_ap = build_factorial_fwd_hooks(
                (layer, head), context_B=False, z_head=None, zero_candidate=False
            )
            hooks_bp = build_factorial_fwd_hooks(
                (layer, head), context_B=True, z_head=None, zero_candidate=False
            )
            zero_a = ios_with(model, tokens, dataset, hooks_a0)
            zero_b = ios_with(model, tokens, dataset, hooks_b0)
            present_a = ios_with(model, tokens, dataset, hooks_ap)
            present_b = ios_with(model, tokens, dataset, hooks_bp)
            intact = present_a - zero_a
            conditional = present_b - zero_b
            gain = conditional - intact

            # 32 cells — cache zero baselines per context
            u_table: Dict[int, CellTable] = {i: {} for i in range(dataset.batch_size)}

            for mask, state in enumerate_extended_fali_cells():
                _, _, z_hybrid = score_pattern_z_pipeline(
                    scores_a, scores_b, v_a, v_b, partition, state
                )
                hooks_inj = build_factorial_fwd_hooks(
                    (layer, head),
                    context_B=bool(state[4]),
                    z_head=z_hybrid,
                    zero_candidate=False,
                )
                metric_inj = ios_with(model, tokens, dataset, hooks_inj)
                base = zero_b if state[4] else zero_a
                u_vec = metric_inj - base
                for i in range(dataset.batch_size):
                    u_val = float(u_vec[i].item())
                    u_table[i][state] = u_val
                    cell_rows.append(
                        {
                            "seed": args.seed,
                            "example_idx": i,
                            "head_name": hname,
                            "layer": layer,
                            "head": head,
                            "T": state[0],
                            "R": state[1],
                            "O": state[2],
                            "V": state[3],
                            "C": state[4],
                            "metric_injected": float(metric_inj[i].item()),
                            "metric_zero": float(base[i].item()),
                            "U": u_val,
                        }
                    )

            if hname in INJECTION_SCOPE_HEADS:
                _, _, z_sample = score_pattern_z_pipeline(
                    scores_a, scores_b, v_a, v_b, partition, (1, 0, 0, 0, 0)
                )
                injection_scope[hname] = run_injection_scope_check(
                    model, tokens, dataset, (layer, head), z_sample, primary_ablate=False
                )
                live_leak[hname] = {
                    "v": run_live_leak_check(
                        model, tokens, dataset, (layer, head), z_sample, primary_ablate=False, perturb="v"
                    ),
                    "q": run_live_leak_check(
                        model, tokens, dataset, (layer, head), z_sample, primary_ablate=False, perturb="q"
                    ),
                    "qv": run_live_leak_check(
                        model, tokens, dataset, (layer, head), z_sample, primary_ablate=False, perturb="qv"
                    ),
                }

            if hname == "L10H2":
                _, _, z_pe = score_pattern_z_pipeline(
                    scores_a, scores_b, v_a, v_b, partition, (0, 0, 0, 0, 0)
                )
                post_end_err = max(
                    post_end_err,
                    run_post_end_causality_check(model, tokens, dataset, (layer, head), z_pe),
                )

            for i in range(dataset.batch_size):
                table = u_table[i]
                u0 = table[(0, 0, 0, 0, 0)]
                u1 = table[(1, 1, 1, 1, 1)]
                intact_i = float(intact[i].item())
                cond_i = float(conditional[i].item())
                gain_i = float(gain[i].item())
                max_endpoint_err = max(
                    max_endpoint_err,
                    abs(u0 - intact_i),
                    abs(u1 - cond_i),
                    abs((u1 - u0) - gain_i),
                )
                diag = owen_diagnostics(table)
                max_owen_res = max(max_owen_res, diag["owen_residual"])
                max_quotient_err = max(max_quotient_err, diag["quotient_block_max"])
                if layer == 9 and hname in SAME_LAYER_HEADS:
                    l9_owen_max[hname] = max(
                        l9_owen_max.get(hname, 0.0),
                        abs(diag["phi_T"]),
                        abs(diag["phi_R"]),
                        abs(diag["phi_O"]),
                        abs(diag["phi_V"]),
                    )

                desc = descriptive_fali_diag(
                    scores_a[i].cpu(),
                    scores_b[i].cpu(),
                    int(dataset.end[i].item()),
                    int(dataset.IO[i].item()),
                )

                hist_row = hist_df[
                    (hist_df["example_idx"] == i) & (hist_df["head_name"] == hname)
                ]
                if len(hist_row) != 1:
                    raise AssertionError(f"Missing historical row for {hname} ex {i}")
                hist_row = hist_row.iloc[0]
                new_h = historical_quotient_cells(table)
                h_errs = compare_historical_cells(new_h, hist_row)
                owen = owen_from_cell_table(table)
                phi_errs = compare_historical_shapley(owen, hist_row)
                max_hist_h = max(max_hist_h, max(h_errs.values()))
                max_hist_phi = max(max_hist_phi, max(phi_errs.values()))

                bridge_rows.append(
                    {
                        "example_idx": i,
                        "head_name": hname,
                        **{f"err_{k}": v for k, v in h_errs.items()},
                        **{f"err_{k}": v for k, v in phi_errs.items()},
                    }
                )

                attr_rows.append(
                    {
                        "seed": args.seed,
                        "example_idx": i,
                        "head_name": hname,
                        "layer": layer,
                        "head": head,
                        "head_role": role,
                        "intact_effect": intact_i,
                        "conditional_effect": cond_i,
                        "conditional_gain": gain_i,
                        "U00000": u0,
                        "U11111": u1,
                        "phi_T": diag["phi_T"],
                        "phi_R": diag["phi_R"],
                        "phi_O": diag["phi_O"],
                        "phi_V": diag["phi_V"],
                        "phi_C": diag["phi_C"],
                        "Phi_SCORE": diag["Phi_SCORE"],
                        "owen_residual": diag["owen_residual"],
                        **desc,
                    }
                )

            head_summaries.append(
                {
                    "head_name": hname,
                    "recon_err_A": err_za,
                    "recon_err_B": err_zb,
                    "score_pat_err_A": err_pa,
                    "score_pat_err_B": err_pb,
                }
            )

        post_snap = snapshot_protected(_REPO_ROOT)
        hash_ok = pre_snap == post_snap

        pd.DataFrame(cell_rows).to_csv(out_dir / "cell_raw.csv", index=False)
        pd.DataFrame(attr_rows).to_csv(out_dir / "attribution_raw.csv", index=False)
        pd.DataFrame(head_summaries).to_csv(out_dir / "head_summary.csv", index=False)
        pd.DataFrame(bridge_rows).to_csv(out_dir / "historical_bridge_comparison.csv", index=False)

        n_cell = len(cell_rows)
        n_attr = len(attr_rows)
        expected_cells = N_HEADS * N_EXPECTED * N_CELLS
        expected_attr = N_HEADS * N_EXPECTED

        checks = {
            "dataset_alignment": align_status == "PASS",
            "cell_row_count": n_cell == expected_cells,
            "attribution_row_count": n_attr == expected_attr,
            "score_pattern": max_score_pat_err <= SCORE_PATTERN_TOL,
            "z_reconstruction": max_z_err <= Z_RECON_TOL,
            "endpoints": max_endpoint_err <= ENDPOINT_TOL,
            "owen_residual": max_owen_res <= OWEN_TOL,
            "quotient_consistency": max_quotient_err <= QUOTIENT_TOL,
            "historical_h_cells": max_hist_h <= HISTORICAL_TOL,
            "historical_shapley": max_hist_phi <= HISTORICAL_TOL,
            "historical_hash_integrity": hash_ok,
            "post_end_causality": post_end_err <= POST_END_TOL,
        }
        for hname in SAME_LAYER_HEADS:
            checks[f"l9_{hname}_score_v"] = l9_score_v_max.get(hname, 0.0) <= SAME_LAYER_SCORE_TOL
            checks[f"l9_{hname}_owen"] = l9_owen_max.get(hname, 0.0) <= SAME_LAYER_OWEN_TOL
        for hname in INJECTION_SCOPE_HEADS:
            sc = injection_scope.get(hname, {})
            checks[f"scope_{hname}_candidate"] = sc.get("candidate_z_error", 999) <= SCORE_PATTERN_TOL
            checks[f"scope_{hname}_noncandidate"] = sc.get("noncandidate_z_error", 999) <= INJECTION_SCOPE_TOL
            lk = live_leak.get(hname, {})
            checks[f"leak_{hname}_v"] = lk.get("v", 999) <= LEAK_TOL
            checks[f"leak_{hname}_q"] = lk.get("q", 999) <= LEAK_TOL
            checks[f"leak_{hname}_qv"] = lk.get("qv", 999) <= LEAK_TOL

        overall = all(checks.values())
        validation.update(
            {
                "overall_pass": overall,
                "checks": checks,
                "n_heads": N_HEADS,
                "n_examples": N_EXPECTED,
                "n_cell_rows": n_cell,
                "n_cell_rows_expected": expected_cells,
                "n_attribution_rows": n_attr,
                "max_score_pattern_error": max_score_pat_err,
                "max_z_reconstruction_error": max_z_err,
                "max_endpoint_error": max_endpoint_err,
                "max_owen_residual": max_owen_res,
                "max_quotient_consistency_error": max_quotient_err,
                "max_historical_h_discrepancy": max_hist_h,
                "max_historical_shapley_discrepancy": max_hist_phi,
                "l9_invariant_score_v_max": l9_score_v_max,
                "l9_invariant_owen_max": l9_owen_max,
                "injection_scope": injection_scope,
                "live_leak": live_leak,
                "post_end_causality_error": post_end_err,
                "historical_hashes_before": pre_snap,
                "historical_hashes_after": post_snap,
                "note": "EF-2 N=8 validation only — not scientific evidence.",
            }
        )

        env = {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformer_lens": pkg_ver("transformer-lens"),
            "transformers": pkg_ver("transformers"),
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "dtype": "float32",
            "git_sha": None,
        }
        try:
            import subprocess

            env["git_sha"] = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT)
                .decode()
                .strip()
            )
        except Exception:
            pass

        run_config = {
            "phase": "EF-2",
            "num_samples": args.num_samples,
            "seed": args.seed,
            "model": "gpt2-small",
            "center_writing_weights": True,
            "dtype": "float32",
            "primary_name_movers": [head_name_from_pair(h) for h in PRIMARY_NAME_MOVERS],
            "candidates": [head_name_from_pair(h) for h in candidates],
            "reference_meta": str(args.reference_meta),
            "historical_factorial": str(args.historical_factorial),
            "duration_sec": time.time() - t0,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
        (out_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
        (out_dir / "validation_report.json").write_text(json.dumps(validation, indent=2) + "\n")

        print(json.dumps(validation, indent=2))
        return 0 if overall else 2

    except Exception as e:
        tb = traceback.format_exc()
        validation["error"] = str(e)
        validation["traceback"] = tb
        validation["overall_pass"] = False
        (out_dir / "validation_report.json").write_text(json.dumps(validation, indent=2) + "\n")
        print(tb, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
