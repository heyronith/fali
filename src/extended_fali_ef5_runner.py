"""
Extended-FALI EF-5 runner: matched-mean / matched-resample primary damage on canonical IOI.
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

from extended_fali_damage import (
    PRIMARY_HEADS,
    build_damage_hooks,
    extract_primary_clean_z,
    verify_hook_preserves_nonprimary_slices,
)
from extended_fali_gpt2 import (
    ENDPOINT_TOL,
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
    align_canonical_dataset_to_reference,
    build_ioi_partition,
    cache_names_filter,
    compute_ios,
    descriptive_fali_diag,
    extract_head_cache,
    max_valid_coord_error,
    owen_diagnostics,
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
from revision_factorial import build_factorial_fwd_hooks
from revision_standard_ioi import StandardIOIDataset

N_CELLS = 32
EF5_HEADS = ("L9H7", "L9H0", "L10H2", "L10H10")


@dataclass
class EF5RunConfig:
    phase: str
    damage_scheme: str
    num_samples: int
    seed: int
    out_dir: Path
    reference_meta: Path
    required_hash_paths: Sequence[str]
    device: str
    num_per_stratum: int = 30
    head_names: Optional[Sequence[str]] = None
    preregistration_sha: Optional[str] = None
    example_indices: Optional[Sequence[int]] = None
    run_smoke_extras: bool = False
    compress_cells: bool = True


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


def ef5_required_hash_paths(repo: Path, seed: int) -> List[str]:
    return [
        f"results/revision_phase4/full/seed_{seed}/factorial_raw.csv",
        f"results/extended_fali/ef4_canonical/seed_{seed}/attribution_raw.csv",
        "results/extended_fali/ef4_canonical/aggregate/hypothesis_results.json",
        "results/extended_fali/ef4_canonical/aggregate/fine_score_gate.json",
    ]


def compute_damage_drop(
    model,
    tokens,
    dataset,
    primary_damage_hooks,
) -> torch.Tensor:
    """damage_drop = IO-S(clean) - IO-S(primary damaged)."""
    with torch.no_grad():
        clean_logits = model(tokens)
    clean_ios = compute_ios(clean_logits, dataset)
    damaged_ios = ios_with(model, tokens, dataset, primary_damage_hooks)
    return clean_ios - damaged_ios


def _dataset_view(
    end: torch.Tensor,
    io: torch.Tensor,
    tokens: torch.Tensor,
    io_token_ids: torch.Tensor,
    s_token_ids: torch.Tensor,
):
    """Minimal dataset view for IO-S computation and partition on a token subset."""

    class View:
        pass

    v = View()
    v.end = end
    v.IO = io
    v.io_token_ids = io_token_ids
    v.s_token_ids = s_token_ids
    v.batch_size = tokens.shape[0]
    v.clean_tokens = tokens
    return v


def run_ef5_seed(cfg: EF5RunConfig, repo_root: Path) -> Dict:
    if cfg.damage_scheme not in ("matched_mean", "matched_resample"):
        raise ValueError(f"EF-5 runner expects matched scheme, got {cfg.damage_scheme}")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    pre_snap = snapshot_paths(repo_root, cfg.required_hash_paths)
    t0 = time.time()
    validation: Dict = {"overall_pass": False, "checks": {}}

    try:
        print(
            f"[{cfg.phase}] scheme={cfg.damage_scheme} seed={cfg.seed} N={cfg.num_samples} ...",
            flush=True,
        )
        model = HookedTransformer.from_pretrained(
            "gpt2-small",
            center_writing_weights=True,
            device=cfg.device,
            dtype=torch.float32,
        )
        model.eval()

        dataset = StandardIOIDataset(model, num_per_stratum=cfg.num_per_stratum, seed=cfg.seed)
        if cfg.num_samples != dataset.num_samples and cfg.example_indices is None:
            raise AssertionError(
                f"N={cfg.num_samples} != dataset.num_samples={dataset.num_samples}"
            )

        tokens_full = dataset.clean_tokens.to(cfg.device)
        meta_full = dataset.metadata_df.copy()

        if cfg.example_indices is not None:
            idx = sorted(int(i) for i in cfg.example_indices)
            tokens = tokens_full[idx]
            meta = meta_full.iloc[idx].reset_index(drop=True)
            meta["example_idx"] = list(range(len(meta)))
            dataset_end = dataset.end[idx].to(cfg.device)
            dataset_io = dataset.IO[idx].to(cfg.device)
            io_tok = dataset.io_token_ids[idx].to(cfg.device)
            s_tok = dataset.s_token_ids[idx].to(cfg.device)
        else:
            idx = list(range(dataset.batch_size))
            tokens = tokens_full
            meta = meta_full
            dataset_end = dataset.end.to(cfg.device)
            dataset_io = dataset.IO.to(cfg.device)
            io_tok = dataset.io_token_ids.to(cfg.device)
            s_tok = dataset.s_token_ids.to(cfg.device)

        batch_size = len(idx)
        ds_view = _dataset_view(dataset_end, dataset_io, tokens, io_tok, s_tok)
        meta.to_csv(cfg.out_dir / "dataset_metadata.csv", index=False)
        ref_meta = pd.read_csv(cfg.reference_meta)
        if cfg.example_indices is None:
            align_status, align_lines = align_canonical_dataset_to_reference(
                meta, ref_meta, token_tensor=dataset.clean_tokens
            )
        else:
            align_status, align_lines = ("PASS", ["Smoke subset: skipped full-cohort alignment"])
        (cfg.out_dir / "dataset_alignment_report.md").write_text(
            f"# Dataset alignment ({cfg.phase} seed {cfg.seed})\n\n" + "\n".join(align_lines) + "\n"
        )
        if align_status != "PASS":
            raise AssertionError("Dataset alignment FAILED")

        partition = build_ioi_partition(ds_view, cfg.device)

        part_diag = partition_diagnostics(partition)

        candidates = all_phase1_candidates()
        if cfg.head_names is not None:
            allowed = set(cfg.head_names)
            candidates = [(l, h) for l, h in candidates if head_name_from_pair((l, h)) in allowed]
        n_heads = len(candidates)
        layers = sorted({l for l, _ in candidates} | {l for l, _ in PRIMARY_NAME_MOVERS})
        names_filter = cache_names_filter(layers)

        print("Cache A (clean)...", flush=True)
        with torch.no_grad():
            _, cache_A = model.run_with_cache(tokens, names_filter=names_filter)

        clean_primary_z = extract_primary_clean_z(cache_A, batch_size)
        damage_hooks, mean_groups, donor_map, donor_audit = build_damage_hooks(
            cfg.damage_scheme,
            clean_z=clean_primary_z,
            meta=meta,
            seed=cfg.seed,
        )

        if mean_groups is not None:
            mean_groups.to_csv(cfg.out_dir / "mean_ablation_groups.csv", index=False)
        if donor_map is not None:
            donor_map.to_csv(cfg.out_dir / "resample_donor_map.csv", index=False)

        print("Cache B (damaged)...", flush=True)
        with model.hooks(fwd_hooks=damage_hooks):
            with torch.no_grad():
                _, cache_B = model.run_with_cache(tokens, names_filter=names_filter)

        damage_drop = compute_damage_drop(model, tokens, ds_view, damage_hooks)

        cell_rows: List[Dict] = []
        attr_rows: List[Dict] = []

        max_score_pat_err = 0.0
        max_z_err = 0.0
        max_endpoint_err = 0.0
        max_owen_res = 0.0
        max_quotient_err = 0.0
        l9_score_v_max: Dict[str, float] = {}
        l9_owen_max: Dict[str, float] = {}
        post_end_err = 0.0
        max_hook_slice_err = 0.0

        if cfg.run_smoke_extras:
            z9 = cache_A["blocks.9.attn.hook_z"].clone()
            max_hook_slice_err = max(
                max_hook_slice_err,
                verify_hook_preserves_nonprimary_slices(z9, damage_hooks, 9, [0, 7]),
            )
            z10 = cache_A["blocks.10.attn.hook_z"].clone()
            max_hook_slice_err = max(
                max_hook_slice_err,
                verify_hook_preserves_nonprimary_slices(z10, damage_hooks, 10, list(range(1, 12))),
            )

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
            err_za = validate_z_reconstruction(p_a_rec, v_a, z_a, dataset_end)
            err_zb = validate_z_reconstruction(p_b_rec, v_b, z_b, dataset_end)
            max_z_err = max(max_z_err, err_za, err_zb)

            if layer == 9 and hname in SAME_LAYER_HEADS:
                sv = max_valid_coord_error(scores_a, scores_b, partition.valid_mask)
                vv = float((v_a - v_b).abs().max().item())
                l9_score_v_max[hname] = max(sv, vv)

            if cfg.run_smoke_extras and hname == "L10H2":
                pass  # post-end check below

            hooks_a0 = build_factorial_fwd_hooks((layer, head), context_B=False, z_head=None, zero_candidate=True)
            hooks_b0 = build_factorial_fwd_hooks((layer, head), context_B=False, z_head=None, zero_candidate=True)
            hooks_ap = build_factorial_fwd_hooks((layer, head), context_B=False, z_head=None, zero_candidate=False)
            hooks_bp = build_factorial_fwd_hooks((layer, head), context_B=False, z_head=None, zero_candidate=False)

            def combine(primary, factorial):
                return list(primary) + list(factorial)

            zero_a = ios_with(model, tokens, ds_view, hooks_a0)
            zero_b = ios_with(model, tokens, ds_view, combine(damage_hooks, hooks_b0))
            present_a = ios_with(model, tokens, ds_view, hooks_ap)
            present_b = ios_with(model, tokens, ds_view, combine(damage_hooks, hooks_bp))
            intact = present_a - zero_a
            conditional = present_b - zero_b
            gain = conditional - intact

            u_table: Dict[int, CellTable] = {i: {} for i in range(batch_size)}

            for _, state in enumerate_extended_fali_cells():
                _, _, z_hybrid = score_pattern_z_pipeline(
                    scores_a, scores_b, v_a, v_b, partition, state
                )
                hooks_inj = build_factorial_fwd_hooks(
                    (layer, head), context_B=False, z_head=z_hybrid, zero_candidate=False
                )
                ctx_primary = damage_hooks if state[4] else []
                metric_inj = ios_with(model, tokens, ds_view, combine(ctx_primary, hooks_inj))
                base_hooks = combine(damage_hooks, hooks_b0) if state[4] else hooks_a0
                base = ios_with(model, tokens, ds_view, base_hooks)
                u_vec = metric_inj - base
                for i in range(batch_size):
                    u_val = float(u_vec[i].item())
                    u_table[i][state] = u_val
                    cell_rows.append(
                        {
                            "seed": cfg.seed,
                            "example_idx": int(meta.iloc[i]["example_idx"]),
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

            if cfg.run_smoke_extras and hname == "L10H2":
                _, _, z_pe = score_pattern_z_pipeline(
                    scores_a, scores_b, v_a, v_b, partition, (0, 0, 0, 0, 0)
                )
                post_end_err = max(
                    post_end_err,
                    run_post_end_causality_check(model, tokens, ds_view, (layer, head), z_pe),
                )

            for i in range(batch_size):
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

                fq = int(dataset_end[i].item())
                tk = int(dataset_io[i].item())
                desc = descriptive_fali_diag(scores_a[i].cpu(), scores_b[i].cpu(), fq, tk)
                clean_attn = float(pat_a[i, fq, tk].item())
                damaged_attn = float(pat_b[i, fq, tk].item())

                attr_row = {
                    "seed": cfg.seed,
                    "example_idx": int(meta.iloc[i]["example_idx"]),
                    "head_name": hname,
                    "layer": layer,
                    "head": head,
                    "head_role": role,
                    "damage_scheme": cfg.damage_scheme,
                    "intact_effect": intact_i,
                    "conditional_effect": cond_i,
                    "conditional_gain": gain_i,
                    "damage_drop": float(damage_drop[i].item()),
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
                if "low_level_template_id" in meta.columns:
                    attr_row["low_level_template_id"] = int(meta.iloc[i]["low_level_template_id"])
                    attr_row["pattern_type"] = meta.iloc[i]["pattern_type"]
                    attr_row["template_stratum"] = meta.iloc[i]["template_stratum"]
                attr_rows.append(attr_row)

        post_snap = snapshot_paths(repo_root, cfg.required_hash_paths)
        hash_ok, hash_checks = verify_historical_hash_integrity(pre_snap, post_snap, cfg.required_hash_paths)

        cell_path = write_cell_raw(cfg.out_dir, cell_rows, cfg.compress_cells)
        pd.DataFrame(attr_rows).to_csv(cfg.out_dir / "attribution_raw.csv", index=False)

        expected_cells = n_heads * batch_size * N_CELLS
        expected_attr = n_heads * batch_size

        checks = {
            "dataset_alignment": True,
            "cell_row_count": len(cell_rows) == expected_cells,
            "attribution_row_count": len(attr_rows) == expected_attr,
            "score_pattern": max_score_pat_err <= SCORE_PATTERN_TOL,
            "z_reconstruction": max_z_err <= Z_RECON_TOL,
            "endpoints": max_endpoint_err <= ENDPOINT_TOL,
            "owen_residual": max_owen_res <= OWEN_TOL,
            "quotient_consistency": max_quotient_err <= QUOTIENT_TOL,
            "historical_hash_integrity": hash_ok,
        }
        if cfg.run_smoke_extras:
            checks["post_end_causality"] = post_end_err <= POST_END_TOL
            checks["replacement_hook_scope"] = max_hook_slice_err <= INJECTION_SCOPE_TOL
        for hname in SAME_LAYER_HEADS:
            checks[f"l9_{hname}_score_v"] = l9_score_v_max.get(hname, 0.0) <= SAME_LAYER_SCORE_TOL
            checks[f"l9_{hname}_owen"] = l9_owen_max.get(hname, 0.0) <= SAME_LAYER_OWEN_TOL

        if donor_audit is not None:
            checks["donor_self_zero"] = donor_audit.get("n_self_donors", -1) == 0
            checks["donor_unmatched_zero"] = donor_audit.get("n_unmatched", -1) == 0

        overall = all(checks.values())
        validation.update(
            {
                "overall_pass": overall,
                "checks": checks,
                "seed": cfg.seed,
                "damage_scheme": cfg.damage_scheme,
                "num_samples": batch_size,
                "batch_partition": {
                    "final_queries": [int(x) for x in dataset_end.tolist()],
                    "target_keys": [int(x) for x in dataset_io.tolist()],
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
                "l9_invariant_score_v_max": l9_score_v_max,
                "l9_invariant_owen_max": l9_owen_max,
                "historical_hash_checks": hash_checks,
                "historical_hashes_before": pre_snap,
                "historical_hashes_after": post_snap,
            }
        )
        if donor_audit is not None:
            validation["donor_audit"] = donor_audit

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
            "damage_scheme": cfg.damage_scheme,
            "num_samples": batch_size,
            "seed": cfg.seed,
            "model": "gpt2-small",
            "dtype": "float32",
            "head_names": list(cfg.head_names) if cfg.head_names else None,
            "preregistration_sha": cfg.preregistration_sha,
            "duration_sec": time.time() - t0,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        (cfg.out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
        (cfg.out_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
        (cfg.out_dir / "validation_report.json").write_text(json.dumps(validation, indent=2) + "\n")
        print(json.dumps({"seed": cfg.seed, "scheme": cfg.damage_scheme, "overall_pass": overall}, indent=2), flush=True)
        return validation

    except Exception as e:
        validation["error"] = str(e)
        validation["traceback"] = traceback.format_exc()
        validation["overall_pass"] = False
        (cfg.out_dir / "validation_report.json").write_text(json.dumps(validation, indent=2) + "\n")
        raise


def compute_zero_damage_drop_seed(
    seed: int,
    device: str,
    repo_root: Path,
    out_csv: Path,
    *,
    num_per_stratum: int = 30,
) -> pd.DataFrame:
    """Forward-only zero damage_drop for reference (no 32-cell game)."""
    from revision_interventions import build_zero_ablation_hooks

    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        center_writing_weights=True,
        device=device,
        dtype=torch.float32,
    )
    model.eval()
    dataset = StandardIOIDataset(model, num_per_stratum=num_per_stratum, seed=seed)
    tokens = dataset.clean_tokens.to(device)
    zero_hooks = build_zero_ablation_hooks(PRIMARY_NAME_MOVERS)
    drop = compute_damage_drop(model, tokens, dataset, zero_hooks)
    meta = dataset.metadata_df.copy()
    rows = []
    for i in range(len(meta)):
        rows.append(
            {
                "seed": seed,
                "example_idx": int(meta.iloc[i]["example_idx"]),
                "damage_scheme": "zero",
                "damage_drop": float(drop[i].item()),
                "low_level_template_id": int(meta.iloc[i]["low_level_template_id"]),
                "pattern_type": meta.iloc[i]["pattern_type"],
                "template_stratum": meta.iloc[i]["template_stratum"],
            }
        )
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df
