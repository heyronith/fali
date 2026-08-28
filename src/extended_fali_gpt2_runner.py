"""
Shared Extended-FALI GPT-2 IOI runner for EF-2 smoke and EF-3 full seeds.
"""

from __future__ import annotations

import gzip
import json
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from transformer_lens import HookedTransformer

from extended_fali_gpt2 import (
    ENDPOINT_TOL,
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
    align_canonical_dataset_to_reference,
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
    snapshot_paths,
    validate_score_pattern,
    validate_z_reconstruction,
    verify_historical_hash_integrity,
)
from extended_fali_interventions import CellTable, enumerate_extended_fali_cells, masked_softmax, score_pattern_z_pipeline
from revision_config import PRIMARY_NAME_MOVERS, all_phase1_candidates, head_name_from_pair, role_for_head
from revision_standard_ioi import StandardIOIDataset
from revision_factorial import build_factorial_fwd_hooks
from revision_interventions import build_zero_ablation_hooks

N_CELLS = 32


@dataclass
class RunConfig:
    phase: str
    num_samples: int
    seed: int
    out_dir: Path
    reference_meta: Path
    historical_factorial: Path
    required_hash_paths: Sequence[str]
    device: str
    run_smoke_extras: bool = False
    compress_cells: bool = False
    dataset_kind: str = "mixed"  # "mixed" | "canonical"
    num_per_stratum: int = 30
    head_names: Optional[Sequence[str]] = None
    preregistration_sha: Optional[str] = None


def resolve_device(requested: Optional[str]) -> str:
    if requested and requested.lower() != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def write_cell_raw(out_dir: Path, rows: List[Dict], compress: bool) -> Path:
    df = pd.DataFrame(rows)
    if compress:
        path = out_dir / "cell_raw.csv.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            df.to_csv(f, index=False)
        return path
    path = out_dir / "cell_raw.csv"
    df.to_csv(path, index=False)
    return path


def run_extended_fali_seed(cfg: RunConfig, repo_root: Path) -> Dict:
    """Run one Extended-FALI seed; return validation report dict."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    pre_snap = snapshot_paths(repo_root, cfg.required_hash_paths)
    t0 = time.time()
    validation: Dict = {"overall_pass": False, "checks": {}}

    try:
        print(f"[{cfg.phase}] seed={cfg.seed} N={cfg.num_samples} loading model...", flush=True)
        model = HookedTransformer.from_pretrained(
            "gpt2-small",
            center_writing_weights=True,
            device=cfg.device,
            dtype="float32",
        )
        model.eval()

        if cfg.dataset_kind == "canonical":
            dataset = StandardIOIDataset(
                model, num_per_stratum=cfg.num_per_stratum, seed=cfg.seed
            )
            if cfg.num_samples != dataset.num_samples:
                raise AssertionError(
                    f"canonical N={cfg.num_samples} != dataset.num_samples={dataset.num_samples}"
                )
            tokens = dataset.clean_tokens.to(cfg.device)
            meta = dataset.metadata_df.copy()
            meta.to_csv(cfg.out_dir / "dataset_metadata.csv", index=False)
            ref_meta = pd.read_csv(cfg.reference_meta)
            align_status, align_lines = align_canonical_dataset_to_reference(
                meta, ref_meta, token_tensor=dataset.clean_tokens
            )
        else:
            from revision_dataset import build_dataset_metadata, make_validated_ioi_dataset

            dataset = make_validated_ioi_dataset(
                model, num_samples=cfg.num_samples, seed=cfg.seed, template_style="mixed"
            )
            tokens = dataset.clean_tokens.to(cfg.device)
            meta = build_dataset_metadata(dataset)
            meta.to_csv(cfg.out_dir / "dataset_metadata.csv", index=False)
            ref_meta = pd.read_csv(cfg.reference_meta)
            align_status, align_lines = align_dataset_to_reference(
                meta, ref_meta, token_tensor=dataset.clean_tokens
            )
        (cfg.out_dir / "dataset_alignment_report.md").write_text(
            f"# Dataset alignment ({cfg.phase} seed {cfg.seed})\n\n" + "\n".join(align_lines) + "\n"
        )
        if align_status != "PASS":
            raise AssertionError("Dataset alignment FAILED")

        if cfg.dataset_kind == "mixed":
            n_a = int((meta["template_id"] == "A").sum())
            n_b = int((meta["template_id"] == "B").sum())
            if cfg.num_samples == 200 and (n_a != 100 or n_b != 100):
                raise AssertionError(f"Expected 100/100 A/B, got {n_a}/{n_b}")
        elif cfg.dataset_kind == "canonical":
            n_strata = meta.groupby(["low_level_template_id", "pattern_type"]).size()
            if not (n_strata == cfg.num_per_stratum).all():
                raise AssertionError(f"Unbalanced canonical strata: {n_strata.to_dict()}")

        partition = build_ioi_partition(dataset, cfg.device)
        part_diag = partition_diagnostics(partition)

        candidates = all_phase1_candidates()
        if cfg.head_names is not None:
            allowed = set(cfg.head_names)
            candidates = [
                (l, h) for l, h in candidates if head_name_from_pair((l, h)) in allowed
            ]
            if len(candidates) != len(cfg.head_names):
                raise AssertionError(f"Head filter mismatch: got {[head_name_from_pair(c) for c in candidates]}")
        n_heads = len(candidates)
        layers = sorted({l for l, _ in candidates} | {l for l, _ in PRIMARY_NAME_MOVERS})
        names_filter = cache_names_filter(layers)
        primary_hooks = build_zero_ablation_hooks(PRIMARY_NAME_MOVERS)

        print("Cache A...", flush=True)
        with torch.no_grad():
            _, cache_A = model.run_with_cache(tokens, names_filter=names_filter)
        print("Cache B...", flush=True)
        with model.hooks(fwd_hooks=primary_hooks):
            with torch.no_grad():
                _, cache_B = model.run_with_cache(tokens, names_filter=names_filter)

        hist_df = pd.read_csv(cfg.historical_factorial)
        cell_rows: List[Dict] = []
        attr_rows: List[Dict] = []
        bridge_rows: List[Dict] = []

        max_score_pat_err = 0.0
        max_z_err = 0.0
        max_endpoint_err = 0.0
        max_owen_res = 0.0
        max_quotient_err = 0.0
        max_hist_h = 0.0
        max_hist_phi = 0.0
        l9_score_v_max: Dict[str, float] = {}
        l9_owen_max: Dict[str, float] = {}
        post_end_err = 0.0

        for layer, head in candidates:
            hname = head_name_from_pair((layer, head))
            role = role_for_head((layer, head))
            print(f"  {hname}...", flush=True)

            scores_a, pat_a, v_a, z_a = extract_head_cache(cache_A, layer, head)
            scores_b, pat_b, v_b, z_b = extract_head_cache(cache_B, layer, head)

            err_pa, mask_pa = validate_score_pattern(scores_a, pat_a, partition)
            err_pb, mask_pb = validate_score_pattern(scores_b, pat_b, partition)
            max_score_pat_err = max(max_score_pat_err, err_pa, err_pb, mask_pa, mask_pb)

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

            hooks_a0 = build_factorial_fwd_hooks((layer, head), context_B=False, z_head=None, zero_candidate=True)
            hooks_b0 = build_factorial_fwd_hooks((layer, head), context_B=True, z_head=None, zero_candidate=True)
            hooks_ap = build_factorial_fwd_hooks((layer, head), context_B=False, z_head=None, zero_candidate=False)
            hooks_bp = build_factorial_fwd_hooks((layer, head), context_B=True, z_head=None, zero_candidate=False)
            zero_a = ios_with(model, tokens, dataset, hooks_a0)
            zero_b = ios_with(model, tokens, dataset, hooks_b0)
            present_a = ios_with(model, tokens, dataset, hooks_ap)
            present_b = ios_with(model, tokens, dataset, hooks_bp)
            intact = present_a - zero_a
            conditional = present_b - zero_b
            gain = conditional - intact

            u_table: Dict[int, CellTable] = {i: {} for i in range(dataset.batch_size)}

            for _, state in enumerate_extended_fali_cells():
                _, _, z_hybrid = score_pattern_z_pipeline(
                    scores_a, scores_b, v_a, v_b, partition, state
                )
                hooks_inj = build_factorial_fwd_hooks(
                    (layer, head), context_B=bool(state[4]), z_head=z_hybrid, zero_candidate=False
                )
                metric_inj = ios_with(model, tokens, dataset, hooks_inj)
                base = zero_b if state[4] else zero_a
                u_vec = metric_inj - base
                for i in range(dataset.batch_size):
                    u_val = float(u_vec[i].item())
                    u_table[i][state] = u_val
                    cell_rows.append(
                        {
                            "seed": cfg.seed,
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

            if cfg.run_smoke_extras and hname in INJECTION_SCOPE_HEADS:
                _, _, z_sample = score_pattern_z_pipeline(
                    scores_a, scores_b, v_a, v_b, partition, (1, 0, 0, 0, 0)
                )
                run_injection_scope_check(
                    model, tokens, dataset, (layer, head), z_sample, primary_ablate=False
                )

            if cfg.run_smoke_extras and hname == "L10H2":
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
                max_quotient_err = max(
                    max_quotient_err,
                    diag["quotient_score_residual"],
                    diag["quotient_value_residual"],
                    diag["quotient_context_residual"],
                )
                if layer == 9 and hname in SAME_LAYER_HEADS:
                    l9_owen_max[hname] = max(
                        l9_owen_max.get(hname, 0.0),
                        abs(diag["phi_T"]),
                        abs(diag["phi_R"]),
                        abs(diag["phi_O"]),
                        abs(diag["phi_V"]),
                    )

                fq = int(dataset.end[i].item())
                tk = int(dataset.IO[i].item())
                desc = descriptive_fali_diag(scores_a[i].cpu(), scores_b[i].cpu(), fq, tk)
                clean_attn = float(pat_a[i, fq, tk].item())
                damaged_attn = float(pat_b[i, fq, tk].item())

                hist_row = hist_df[(hist_df["example_idx"] == i) & (hist_df["head_name"] == hname)]
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
                    {"example_idx": i, "head_name": hname, **{f"err_{k}": v for k, v in {**h_errs, **phi_errs}.items()}}
                )

                attr_row = {
                    "seed": cfg.seed,
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
                    "clean_target_attn": clean_attn,
                    "damaged_target_attn": damaged_attn,
                    "delta_target_attn": damaged_attn - clean_attn,
                    **desc,
                }
                if "template_id" in meta.columns:
                    attr_row["template_id"] = meta.iloc[i]["template_id"]
                if "low_level_template_id" in meta.columns:
                    attr_row["low_level_template_id"] = int(meta.iloc[i]["low_level_template_id"])
                    attr_row["pattern_type"] = meta.iloc[i]["pattern_type"]
                    attr_row["template_stratum"] = meta.iloc[i]["template_stratum"]
                attr_rows.append(attr_row)

        post_snap = snapshot_paths(repo_root, cfg.required_hash_paths)
        hash_ok, hash_checks = verify_historical_hash_integrity(pre_snap, post_snap, cfg.required_hash_paths)

        cell_path = write_cell_raw(cfg.out_dir, cell_rows, cfg.compress_cells)
        pd.DataFrame(attr_rows).to_csv(cfg.out_dir / "attribution_raw.csv", index=False)
        pd.DataFrame(bridge_rows).to_csv(cfg.out_dir / "historical_bridge_comparison.csv", index=False)

        expected_cells = n_heads * cfg.num_samples * N_CELLS
        expected_attr = n_heads * cfg.num_samples

        checks = {
            "dataset_alignment": True,
            "cell_row_count": len(cell_rows) == expected_cells,
            "attribution_row_count": len(attr_rows) == expected_attr,
            "score_pattern": max_score_pat_err <= SCORE_PATTERN_TOL,
            "z_reconstruction": max_z_err <= Z_RECON_TOL,
            "endpoints": max_endpoint_err <= ENDPOINT_TOL,
            "owen_residual": max_owen_res <= OWEN_TOL,
            "quotient_consistency": max_quotient_err <= QUOTIENT_TOL,
            "historical_h_cells": max_hist_h <= HISTORICAL_TOL,
            "historical_shapley": max_hist_phi <= HISTORICAL_TOL,
            "historical_hash_integrity": hash_ok,
        }
        if cfg.run_smoke_extras:
            checks["post_end_causality"] = post_end_err <= POST_END_TOL
        for hname in SAME_LAYER_HEADS:
            checks[f"l9_{hname}_score_v"] = l9_score_v_max.get(hname, 0.0) <= SAME_LAYER_SCORE_TOL
            checks[f"l9_{hname}_owen"] = l9_owen_max.get(hname, 0.0) <= SAME_LAYER_OWEN_TOL

        overall = all(checks.values())
        validation.update(
            {
                "overall_pass": overall,
                "checks": checks,
                "seed": cfg.seed,
                "num_samples": cfg.num_samples,
                "batch_partition": {
                    "final_queries": [int(x) for x in dataset.end.tolist()],
                    "target_keys": [int(x) for x in dataset.IO.tolist()],
                    "valid_lengths": [int(x) for x in (dataset.end + 1).tolist()],
                    **part_diag,
                },
                "n_cell_rows": len(cell_rows),
                "n_cell_rows_expected": expected_cells,
                "n_attribution_rows": len(attr_rows),
                "cell_raw_path": str(cell_path.name),
                "max_score_pattern_error": max_score_pat_err,
                "max_z_reconstruction_error": max_z_err,
                "max_endpoint_error": max_endpoint_err,
                "max_owen_residual": max_owen_res,
                "max_quotient_consistency_error": max_quotient_err,
                "max_historical_h_discrepancy": max_hist_h,
                "max_historical_shapley_discrepancy": max_hist_phi,
                "l9_invariant_score_v_max": l9_score_v_max,
                "l9_invariant_owen_max": l9_owen_max,
                "historical_hash_checks": hash_checks,
                "historical_hashes_before": pre_snap,
                "historical_hashes_after": post_snap,
            }
        )

        env = {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformer_lens": pkg_ver("transformer-lens"),
            "transformers": pkg_ver("transformers"),
            "device": cfg.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "dtype": "float32",
        }
        run_config = {
            "phase": cfg.phase,
            "num_samples": cfg.num_samples,
            "seed": cfg.seed,
            "model": "gpt2-small",
            "dtype": "float32",
            "dataset_kind": cfg.dataset_kind,
            "head_names": list(cfg.head_names) if cfg.head_names else None,
            "preregistration_sha": cfg.preregistration_sha,
            "duration_sec": time.time() - t0,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        (cfg.out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
        (cfg.out_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
        (cfg.out_dir / "validation_report.json").write_text(json.dumps(validation, indent=2) + "\n")
        print(json.dumps({"seed": cfg.seed, "overall_pass": overall}, indent=2), flush=True)
        return validation

    except Exception as e:
        validation["error"] = str(e)
        validation["traceback"] = traceback.format_exc()
        validation["overall_pass"] = False
        (cfg.out_dir / "validation_report.json").write_text(json.dumps(validation, indent=2) + "\n")
        raise
