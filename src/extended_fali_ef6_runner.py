"""
EF-6 Pythia-160M induction transfer runner (smoke, discovery, confirmation).
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

import numpy as np
import pandas as pd
import torch
from transformer_lens import HookedTransformer

from extended_fali_ef6_analysis import (
    ATTN_THRESHOLD,
    N_BOOTSTRAP,
    PairCandidate,
    bootstrap_offset_balanced,
    candidate_pair_eligible,
    evaluate_scalar_gate,
    head_name,
    load_selection_lock,
    observed_offset_balanced_mean,
    offset_means,
    percentile_ci,
    primary_eligible,
    select_head_pair,
    write_selection_lock,
)
from extended_fali_gpt2 import (
    ENDPOINT_TOL,
    INJECTION_SCOPE_TOL,
    LEAK_TOL,
    OWEN_TOL,
    QUOTIENT_TOL,
    SCORE_PATTERN_TOL,
    Z_RECON_TOL,
    cache_names_filter,
    extract_head_cache,
    max_z_error_on_prompt,
    owen_diagnostics,
    partition_diagnostics,
    validate_score_pattern,
    validate_z_reconstruction,
)
from extended_fali_induction import (
    CONFIRMATION_SEEDS,
    DISCOVERY_SEED,
    EXAMPLES_PER_OFFSET,
    QUERY_OFFSETS,
    InductionDataset,
    generate_induction_metadata,
    metadata_to_tensors,
    save_dataset_artifacts,
    tensor_sha256,
    validate_induction_metadata,
)
from extended_fali_interventions import CellTable, enumerate_extended_fali_cells, masked_softmax, score_pattern_z_pipeline
from historical_integrity_audit import audit_ef5_historical_integrity
from revision_factorial import make_full_z_injection_hook, merge_hook_lists
from revision_interventions import build_zero_ablation_hooks

MODEL_ID = "EleutherAI/pythia-160m"
PREREG_SHA = "3a095b4a73b4e82c2c7003dfeb6b64c6a0aeb301"
N_LAYERS = 12
N_HEADS = 12
D_MODEL = 768
N_CELLS = 32
SMOKE_N = 8
SMOKE_PRIMARY = (0, 0)
SMOKE_CANDIDATE = (1, 0)
BATCH_CHUNK = 32

Head = Tuple[int, int]


@dataclass
class EF6RunConfig:
    phase: str
    seed: int
    out_dir: Path
    device: str
    preregistration_sha: str = PREREG_SHA
    repo_root: Optional[Path] = None
    compress_cells: bool = True
    run_smoke: bool = False
    selection_lock_path: Optional[Path] = None


def pkg_ver(name: str) -> Optional[str]:
    try:
        return pkg_version(name)
    except PackageNotFoundError:
        return None


def resolve_device(requested: Optional[str]) -> str:
    if requested and requested.lower() not in ("auto", ""):
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_model_revision(repo: Path, out_dir: Path) -> str:
    rev_path = out_dir / "model_revision.json"
    if rev_path.exists():
        return json.loads(rev_path.read_text())["resolved_revision_sha"]
    from huggingface_hub import HfApi

    info = HfApi().model_info(MODEL_ID)
    payload = {
        "repo_id": MODEL_ID,
        "resolved_revision_sha": info.sha,
        "transformer_lens_version": pkg_ver("transformer-lens"),
        "transformers_version": pkg_ver("transformers"),
        "torch_version": torch.__version__,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    rev_path.write_text(json.dumps(payload, indent=2) + "\n")
    return info.sha


def load_pythia(device: str, revision: str) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(
        MODEL_ID,
        center_writing_weights=True,
        device=device,
        dtype=torch.float32,
        revision=revision,
    )
    model.eval()
    assert model.cfg.n_layers == N_LAYERS
    assert model.cfg.n_heads == N_HEADS
    assert model.cfg.d_model == D_MODEL
    return model


def target_logprob(
    model: HookedTransformer,
    tokens: torch.Tensor,
    query_pos: torch.Tensor,
    target_ids: torch.Tensor,
    hooks: Optional[List] = None,
) -> torch.Tensor:
    with torch.no_grad():
        if hooks:
            with model.hooks(fwd_hooks=hooks):
                logits = model(tokens)
        else:
            logits = model(tokens)
    log_probs = torch.log_softmax(logits, dim=-1)
    batch = tokens.shape[0]
    bi = torch.arange(batch, device=tokens.device)
    return log_probs[bi, query_pos, target_ids]


def build_ef6_factorial_hooks(
    candidate: Head,
    primary: Head,
    *,
    context_b: bool,
    z_head: Optional[torch.Tensor],
    zero_candidate: bool,
) -> List:
    c_layer, c_head = candidate
    p_layer, p_head = primary
    hooks: List = []
    if context_b:
        hooks.append(
            make_full_z_injection_hook(
                p_layer,
                p_head,
                None,
                zero_heads=[p_head],
                zero_candidate=True,
            )
        )
    hooks.append(
        make_full_z_injection_hook(
            c_layer,
            c_head,
            z_head,
            zero_heads=(),
            zero_candidate=zero_candidate,
        )
    )
    return merge_hook_lists(hooks)


def _chunked_metric(
    model: HookedTransformer,
    dataset: InductionDataset,
    device: str,
    hooks: Optional[List] = None,
    *,
    repeated: bool = True,
) -> np.ndarray:
    tokens_all = dataset.repeated_tokens if repeated else dataset.control_tokens
    n = dataset.batch_size
    out = np.zeros(n, dtype=np.float64)
    for start in range(0, n, BATCH_CHUNK):
        end = min(start + BATCH_CHUNK, n)
        tokens = tokens_all[start:end].to(device)
        q = dataset.query_pos[start:end].to(device)
        tid = dataset.target_token_id[start:end].to(device)
        out[start:end] = target_logprob(model, tokens, q, tid, hooks).cpu().numpy()
    return out


def compute_induction_advantage(model, dataset, device) -> np.ndarray:
    rep = _chunked_metric(model, dataset, device, repeated=True)
    ctl = _chunked_metric(model, dataset, device, repeated=False)
    return rep - ctl


def compute_all_head_clean_attention(
    model: HookedTransformer,
    dataset: InductionDataset,
    device: str,
) -> pd.DataFrame:
    layers = list(range(N_LAYERS))
    names = cache_names_filter(layers)
    rows = []
    n = dataset.batch_size
    for start in range(0, n, BATCH_CHUNK):
        end = min(start + BATCH_CHUNK, n)
        tokens = dataset.repeated_tokens[start:end].to(device)
        q = dataset.query_pos[start:end]
        k = dataset.target_key[start:end]
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=names)
        bi = torch.arange(end - start)
        q_local = q - 0  # already absolute positions in full sequence
        for layer in range(N_LAYERS):
            pat = cache[f"blocks.{layer}.attn.hook_pattern"].cpu()
            for head in range(N_HEADS):
                attn = pat[bi, head, q_local, k].numpy()
                for i, val in enumerate(attn):
                    ex = start + i
                    rows.append(
                        {
                            "example_idx": int(dataset.metadata.iloc[ex]["example_idx"]),
                            "layer": layer,
                            "head": head,
                            "head_name": head_name(layer, head),
                            "clean_induction_attention": float(val),
                        }
                    )
    return pd.DataFrame(rows)


def compute_head_effect(
    model: HookedTransformer,
    dataset: InductionDataset,
    device: str,
    heads: Sequence[Head],
    *,
    primary: Optional[Head] = None,
) -> pd.DataFrame:
    clean = _chunked_metric(model, dataset, device)
    rows = []
    for layer, head in heads:
        ablate = [((layer, head))]
        if primary is not None:
            ablate = [primary, (layer, head)]
        hooks = build_zero_ablation_hooks(ablate)
        damaged = _chunked_metric(model, dataset, device, hooks=hooks)
        effect = clean - damaged
        meta = dataset.metadata
        for i in range(len(effect)):
            rows.append(
                {
                    "example_idx": int(meta.iloc[i]["example_idx"]),
                    "layer": layer,
                    "head": head,
                    "head_name": head_name(layer, head),
                    "primary_layer": None if primary is None else primary[0],
                    "primary_head": None if primary is None else primary[1],
                    "effect": float(effect[i]),
                    "query_offset": int(meta.iloc[i]["query_offset"]),
                }
            )
    return pd.DataFrame(rows)


def _effect_summary(df: pd.DataFrame, value_col: str = "effect") -> pd.DataFrame:
    return (
        df.groupby(["layer", "head", "head_name"], as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: "balanced_mean"})
    )


def run_discovery_pair_selection(
    model,
    dataset: InductionDataset,
    device: str,
    attn_df: pd.DataFrame,
    intact_df: pd.DataFrame,
) -> Tuple[Optional[PairCandidate], pd.DataFrame, List[PairCandidate], int]:
    """Discovery pair selection without phi_*."""
    offsets = dataset.metadata["query_offset"].values
    attn_heads = {}
    for (layer, head), grp in attn_df.groupby(["layer", "head"]):
        attn_heads[(int(layer), int(head))] = float(grp["clean_induction_attention"].mean())

    candidate_heads = [h for h, v in attn_heads.items() if v >= ATTN_THRESHOLD]
    pair_rows: List[PairCandidate] = []
    pairs_evaluated = 0

    def primary_intact_balanced(p_layer: int, p_head: int) -> float:
        sub = intact_df[
            (intact_df["layer"].astype(int) == p_layer) & (intact_df["head"].astype(int) == p_head)
        ]
        return observed_offset_balanced_mean(sub["effect"].values, sub["query_offset"].values)

    primaries = []
    for (pl, ph), grp in intact_df.groupby(["layer", "head"]):
        pl_i, ph_i = int(pl), int(ph)
        if attn_heads.get((pl_i, ph_i), 0) < ATTN_THRESHOLD:
            continue
        vals = grp["effect"].values
        grp_offsets = grp["query_offset"].values
        boot = bootstrap_offset_balanced(vals, grp_offsets, scope=f"intact_{pl_i}_{ph_i}", metric="intact")
        ci_lo, ci_hi = percentile_ci(boot)
        off_m = offset_means(vals, grp_offsets)
        later = any(
            cl > pl_i and attn_heads.get((cl, ch), 0) >= ATTN_THRESHOLD
            for cl in range(N_LAYERS)
            for ch in range(N_HEADS)
        )
        ok, _ = primary_eligible(
            clean_attention=attn_heads.get((pl_i, ph_i), 0),
            intact_ci_low=ci_lo,
            intact_ci_high=ci_hi,
            offset_intact_means=off_m,
            has_later_attention_head=later,
        )
        if ok:
            primaries.append((pl_i, ph_i))

    for primary in primaries:
        pl, ph = primary
        later_heads = [
            h
            for h in candidate_heads
            if h[0] > pl and attn_heads.get(h, 0) >= ATTN_THRESHOLD
        ]
        if not later_heads:
            continue
        cond_df = compute_head_effect(model, dataset, device, later_heads, primary=primary)
        for (cl, ch), grp in cond_df.groupby(["layer", "head"]):
            pairs_evaluated += 1
            cl_int, ch_int = int(cl), int(ch)
            intact_sub = intact_df[
                (intact_df["layer"].astype(int) == cl_int) & (intact_df["head"].astype(int) == ch_int)
            ]
            merged = grp.merge(
                intact_sub[["example_idx", "effect"]].rename(columns={"effect": "intact_effect"}),
                on="example_idx",
                how="inner",
            )
            if len(merged) != len(offsets):
                pair_rows.append(
                    PairCandidate(
                        primary_layer=pl,
                        primary_head=ph,
                        candidate_layer=cl_int,
                        candidate_head=ch_int,
                        conditional_gain_balanced=float("nan"),
                        conditional_effect_balanced=float("nan"),
                        candidate_clean_attention=attn_heads.get((cl_int, ch_int), 0),
                        primary_intact_effect=primary_intact_balanced(pl, ph),
                        eligible=False,
                        reason="merge_empty",
                    )
                )
                continue
            cond_vals = merged["effect"].values
            intact_vals = merged["intact_effect"].values
            pair_offsets = merged["query_offset"].values
            gain_vals = cond_vals - intact_vals
            cond_boot = bootstrap_offset_balanced(
                cond_vals, pair_offsets, scope=f"cond_{pl}_{ph}_{cl}_{ch}", metric="conditional"
            )
            gain_boot = bootstrap_offset_balanced(
                gain_vals, pair_offsets, scope=f"gain_{pl}_{ph}_{cl}_{ch}", metric="gain"
            )
            c_lo, c_hi = percentile_ci(cond_boot)
            g_lo, g_hi = percentile_ci(gain_boot)
            off_cond = offset_means(cond_vals, pair_offsets)
            off_gain = offset_means(gain_vals, pair_offsets)
            elig, reason = candidate_pair_eligible(
                primary_layer=pl,
                candidate_layer=int(cl),
                cond_ci_low=c_lo,
                cond_ci_high=c_hi,
                gain_ci_low=g_lo,
                gain_ci_high=g_hi,
                offset_cond_means=off_cond,
                offset_gain_means=off_gain,
            )
            pc = PairCandidate(
                primary_layer=pl,
                primary_head=ph,
                candidate_layer=int(cl),
                candidate_head=int(ch),
                conditional_gain_balanced=observed_offset_balanced_mean(gain_vals, pair_offsets),
                conditional_effect_balanced=observed_offset_balanced_mean(cond_vals, pair_offsets),
                candidate_clean_attention=attn_heads.get((int(cl), int(ch)), 0),
                primary_intact_effect=primary_intact_balanced(pl, ph),
                eligible=elig,
                reason=reason,
            )
            pair_rows.append(pc)

    selected = select_head_pair(pair_rows)
    pairwise_csv = pd.DataFrame(
        [
            {
                "primary_layer": p.primary_layer,
                "primary_head": p.primary_head,
                "candidate_layer": p.candidate_layer,
                "candidate_head": p.candidate_head,
                "conditional_gain": p.conditional_gain_balanced,
                "conditional_effect": p.conditional_effect_balanced,
                "eligible": p.eligible,
                "reason": p.reason,
            }
            for p in pair_rows
        ]
    )
    return selected, pairwise_csv, pair_rows, pairs_evaluated


def run_extended_fali_confirmation(
    model: HookedTransformer,
    dataset: InductionDataset,
    device: str,
    primary: Head,
    candidate: Head,
    seed: int,
) -> Tuple[List[Dict], List[Dict], Dict]:
    partition = dataset.partition(device)
    part_diag = partition_diagnostics(partition)
    c_layer, c_head = candidate
    p_layer, p_head = primary
    layers = sorted({p_layer, c_layer})
    names = cache_names_filter(layers)
    primary_hooks = build_zero_ablation_hooks([primary])
    tokens = dataset.repeated_tokens.to(device)

    with torch.no_grad():
        _, cache_a = model.run_with_cache(tokens, names_filter=names)
    with model.hooks(fwd_hooks=primary_hooks):
        with torch.no_grad():
            _, cache_b = model.run_with_cache(tokens, names_filter=names)

    scores_a, pat_a, v_a, z_a = extract_head_cache(cache_a, c_layer, c_head)
    scores_b, pat_b, v_b, z_b = extract_head_cache(cache_b, c_layer, c_head)

    max_score_pat = 0.0
    max_z = 0.0
    max_endpoint = 0.0
    max_owen = 0.0
    max_quotient = 0.0

    err_pa, mask_pa = validate_score_pattern(scores_a, pat_a, partition)
    err_pb, mask_pb = validate_score_pattern(scores_b, pat_b, partition)
    max_score_pat = max(max_score_pat, err_pa, err_pb, mask_pa, mask_pb)

    p_a_rec = masked_softmax(scores_a.masked_fill(~partition.valid_mask, float("-inf")), partition.valid_mask)
    p_b_rec = masked_softmax(scores_b.masked_fill(~partition.valid_mask, float("-inf")), partition.valid_mask)
    max_z = max(
        max_z,
        validate_z_reconstruction(p_a_rec, v_a, z_a, dataset.query_pos),
        validate_z_reconstruction(p_b_rec, v_b, z_b, dataset.query_pos),
    )

    hooks_a0 = build_ef6_factorial_hooks(candidate, primary, context_b=False, z_head=None, zero_candidate=True)
    hooks_b0 = build_ef6_factorial_hooks(candidate, primary, context_b=True, z_head=None, zero_candidate=True)
    hooks_ap = build_ef6_factorial_hooks(candidate, primary, context_b=False, z_head=None, zero_candidate=False)
    hooks_bp = build_ef6_factorial_hooks(candidate, primary, context_b=True, z_head=None, zero_candidate=False)

    zero_a = target_logprob(model, tokens, dataset.query_pos.to(device), dataset.target_token_id.to(device), hooks_a0)
    zero_b = target_logprob(model, tokens, dataset.query_pos.to(device), dataset.target_token_id.to(device), hooks_b0)
    present_a = target_logprob(model, tokens, dataset.query_pos.to(device), dataset.target_token_id.to(device), hooks_ap)
    present_b = target_logprob(model, tokens, dataset.query_pos.to(device), dataset.target_token_id.to(device), hooks_bp)
    intact = present_a - zero_a
    conditional = present_b - zero_b
    gain = conditional - intact

    cell_rows: List[Dict] = []
    attr_rows: List[Dict] = []
    meta = dataset.metadata
    u_tables: Dict[int, CellTable] = {i: {} for i in range(dataset.batch_size)}

    for _, state in enumerate_extended_fali_cells():
        _, _, z_hybrid = score_pattern_z_pipeline(
            scores_a, scores_b, v_a, v_b, partition, state
        )
        hooks_inj = build_ef6_factorial_hooks(
            candidate,
            primary,
            context_b=bool(state[4]),
            z_head=z_hybrid,
            zero_candidate=False,
        )
        metric_inj = target_logprob(
            model,
            tokens,
            dataset.query_pos.to(device),
            dataset.target_token_id.to(device),
            hooks_inj,
        )
        base = zero_b if state[4] else zero_a
        u_vec = metric_inj - base
        for i in range(dataset.batch_size):
            u_val = float(u_vec[i].item())
            u_tables[i][state] = u_val
            cell_rows.append(
                {
                    "seed": seed,
                    "example_idx": int(meta.iloc[i]["example_idx"]),
                    "query_offset": int(meta.iloc[i]["query_offset"]),
                    "layer": c_layer,
                    "head": c_head,
                    "head_name": head_name(c_layer, c_head),
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

    for i in range(dataset.batch_size):
        u_table = u_tables[i]
        u0 = u_table[(0, 0, 0, 0, 0)]
        u1 = u_table[(1, 1, 1, 1, 1)]
        intact_i = float(intact[i].item())
        cond_i = float(conditional[i].item())
        gain_i = float(gain[i].item())
        max_endpoint = max(max_endpoint, abs(u0 - intact_i), abs(u1 - cond_i), abs((u1 - u0) - gain_i))
        diag = owen_diagnostics(u_table)
        max_owen = max(max_owen, diag["owen_residual"])
        max_quotient = max(
            max_quotient,
            diag["quotient_score_residual"],
            diag["quotient_value_residual"],
            diag["quotient_context_residual"],
        )
        fq = int(dataset.query_pos[i].item())
        tk = int(dataset.target_key[i].item())
        attr_rows.append(
            {
                "seed": seed,
                "example_idx": int(meta.iloc[i]["example_idx"]),
                "query_offset": int(meta.iloc[i]["query_offset"]),
                "head_name": head_name(c_layer, c_head),
                "layer": c_layer,
                "head": c_head,
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
                "clean_target_attn": float(pat_a[i, fq, tk].item()),
                "damaged_target_attn": float(pat_b[i, fq, tk].item()),
                "delta_target_attn": float(pat_b[i, fq, tk].item() - pat_a[i, fq, tk].item()),
            }
        )

    validation = {
        "max_score_pattern_error": max_score_pat,
        "max_z_reconstruction_error": max_z,
        "max_endpoint_error": max_endpoint,
        "max_owen_residual": max_owen,
        "max_quotient_consistency_error": max_quotient,
        "partition": part_diag,
        "checks": {
            "score_pattern": max_score_pat <= SCORE_PATTERN_TOL,
            "z_reconstruction": max_z <= Z_RECON_TOL,
            "endpoints": max_endpoint <= ENDPOINT_TOL,
            "owen_residual": max_owen <= OWEN_TOL,
            "quotient_consistency": max_quotient <= QUOTIENT_TOL,
        },
    }
    validation["overall_pass"] = all(validation["checks"].values())
    return cell_rows, attr_rows, validation


def write_cell_raw(out_dir: Path, rows: List[Dict], compress: bool) -> Path:
    df = pd.DataFrame(rows)
    if compress:
        path = out_dir / "cell_raw.csv.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            df.to_csv(f, index=False)
    else:
        path = out_dir / "cell_raw.csv"
        df.to_csv(path, index=False)
    return path


def _write_env(out_dir: Path, device: str) -> None:
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
        "modal_gpu": "L4",
    }
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n")


def run_ef6_smoke(cfg: EF6RunConfig) -> Dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    repo = cfg.repo_root or cfg.out_dir.parents[3]
    ef6_root = repo / "results/extended_fali/ef6_pythia_induction"
    revision = resolve_model_revision(repo, ef6_root)
    model = load_pythia(cfg.device, revision)
    meta = generate_induction_metadata(seed=cfg.seed, smoke_deterministic=True, smoke_n=SMOKE_N)
    dataset = metadata_to_tensors(meta)
    save_dataset_artifacts(dataset, cfg.out_dir)
    cell_rows, attr_rows, val = run_extended_fali_confirmation(
        model, dataset, cfg.device, SMOKE_PRIMARY, SMOKE_CANDIDATE, cfg.seed
    )
    write_cell_raw(cfg.out_dir, cell_rows, compress=False)
    pd.DataFrame(attr_rows).to_csv(cfg.out_dir / "attribution_raw.csv", index=False)
    val["phase"] = "smoke"
    val["primary"] = head_name(*SMOKE_PRIMARY)
    val["candidate"] = head_name(*SMOKE_CANDIDATE)
    (cfg.out_dir / "validation_report.json").write_text(json.dumps(val, indent=2) + "\n")
    run_cfg = {
        "phase": "smoke",
        "seed": cfg.seed,
        "model": MODEL_ID,
        "model_revision": revision,
        "preregistration_sha": cfg.preregistration_sha,
        "primary": SMOKE_PRIMARY,
        "candidate": SMOKE_CANDIDATE,
        "n_examples": SMOKE_N,
    }
    (cfg.out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2) + "\n")
    _write_env(cfg.out_dir, cfg.device)
    return val


def run_ef6_discovery(cfg: EF6RunConfig) -> Dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    repo = cfg.repo_root or cfg.out_dir.parents[3]
    ef6_root = repo / "results/extended_fali/ef6_pythia_induction"
    revision = resolve_model_revision(repo, ef6_root)
    model = load_pythia(cfg.device, revision)
    meta = generate_induction_metadata(seed=DISCOVERY_SEED)
    validate_induction_metadata(meta)
    dataset = metadata_to_tensors(meta)
    ds_info = save_dataset_artifacts(dataset, cfg.out_dir)
    offsets = dataset.metadata["query_offset"].values

    adv = compute_induction_advantage(model, dataset, cfg.device)
    g0 = evaluate_scalar_gate("G0_discovery", adv, np.full(len(adv), DISCOVERY_SEED), offsets, discovery=True, positive=True, scope="g0")
    behavior_df = pd.DataFrame(
        {
            "example_idx": dataset.metadata["example_idx"],
            "query_offset": offsets,
            "induction_advantage": adv,
        }
    )
    behavior_df.to_csv(cfg.out_dir / "behavior_results.csv", index=False)

    if g0.status != "CONFIRMED":
        report = {
            "G0_discovery": g0.__dict__,
            "EF6_GATE": "FAIL",
            "fine_attributions_used_for_selection": False,
            "stop_reason": "G0_discovery_failed",
        }
        (cfg.out_dir / "selection_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        return report

    attn_df = compute_all_head_clean_attention(model, dataset, cfg.device)
    attn_df.to_csv(cfg.out_dir / "all_head_induction_scores.csv", index=False)
    intact_heads = [(l, h) for l in range(N_LAYERS) for h in range(N_HEADS)]
    intact_df = compute_head_effect(model, dataset, cfg.device, intact_heads)
    intact_df.to_csv(cfg.out_dir / "all_head_intact_effects.csv", index=False)

    selected, pairwise, pair_rows, n_pairs = run_discovery_pair_selection(
        model, dataset, cfg.device, attn_df, intact_df
    )
    pairwise.to_csv(cfg.out_dir / "pairwise_recruitment.csv", index=False)

    eligible_primaries = len({(p.primary_layer, p.primary_head) for p in pair_rows if p.eligible})
    eligible_pairs = sum(1 for p in pair_rows if p.eligible)

    report = {
        "dataset_sha256": ds_info["tensor_sha256"],
        "G0_discovery": g0.__dict__,
        "fine_attributions_used_for_selection": False,
        "pairs_evaluated": n_pairs,
        "eligible_pairs": eligible_pairs,
        "eligible_primary_count": eligible_primaries,
    }
    if selected is None:
        report["EF6_GATE"] = "FAIL"
        report["stop_reason"] = "no_eligible_pair"
    else:
        report["selected_primary"] = {
            "layer": selected.primary_layer,
            "head": selected.primary_head,
            "head_name": head_name(selected.primary_layer, selected.primary_head),
            "intact_effect": selected.primary_intact_effect,
        }
        report["selected_candidate"] = {
            "layer": selected.candidate_layer,
            "head": selected.candidate_head,
            "head_name": head_name(selected.candidate_layer, selected.candidate_head),
            "conditional_gain": selected.conditional_gain_balanced,
            "conditional_effect": selected.conditional_effect_balanced,
            "clean_attention": selected.candidate_clean_attention,
        }
        lock_path = ef6_root / "aggregate" / "selection_lock.json"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        write_selection_lock(
            lock_path,
            selection=selected,
            preregistration_sha=cfg.preregistration_sha,
            dataset_sha=ds_info["tensor_sha256"],
        )
    (cfg.out_dir / "selection_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    run_cfg = {
        "phase": "discovery",
        "seed": DISCOVERY_SEED,
        "model": MODEL_ID,
        "model_revision": revision,
        "preregistration_sha": cfg.preregistration_sha,
    }
    (cfg.out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2) + "\n")
    _write_env(cfg.out_dir, cfg.device)
    return report


def run_ef6_confirmation(cfg: EF6RunConfig) -> Dict:
    if cfg.selection_lock_path is None:
        raise ValueError("selection_lock_path required for confirmation")
    lock = load_selection_lock(cfg.selection_lock_path)
    primary = (lock["primary"]["layer"], lock["primary"]["head"])
    candidate = (lock["candidate"]["layer"], lock["candidate"]["head"])
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    repo = cfg.repo_root or cfg.out_dir.parents[4]
    ef6_root = repo / "results/extended_fali/ef6_pythia_induction"
    revision = resolve_model_revision(repo, ef6_root)
    stored_rev = json.loads((ef6_root / "model_revision.json").read_text())["resolved_revision_sha"]
    if revision != stored_rev:
        raise RuntimeError("Model revision changed between runs — STOP")
    model = load_pythia(cfg.device, revision)
    meta = generate_induction_metadata(seed=cfg.seed)
    dataset = metadata_to_tensors(meta)
    save_dataset_artifacts(dataset, cfg.out_dir)
    adv = compute_induction_advantage(model, dataset, cfg.device)
    primary_hooks = build_zero_ablation_hooks([primary])
    clean_rep = _chunked_metric(model, dataset, cfg.device, repeated=True)
    damaged_rep = _chunked_metric(model, dataset, cfg.device, hooks=primary_hooks, repeated=True)
    primary_drop = clean_rep - damaged_rep
    pd.DataFrame(
        {
            "seed": cfg.seed,
            "example_idx": dataset.metadata["example_idx"],
            "query_offset": dataset.metadata["query_offset"],
            "induction_advantage": adv,
            "primary_drop": primary_drop,
        }
    ).to_csv(cfg.out_dir / "behavior_results.csv", index=False)
    cell_rows, attr_rows, val = run_extended_fali_confirmation(
        model, dataset, cfg.device, primary, candidate, cfg.seed
    )
    write_cell_raw(cfg.out_dir, cell_rows, cfg.compress_cells)
    pd.DataFrame(attr_rows).to_csv(cfg.out_dir / "attribution_raw.csv", index=False)
    expected_cells = dataset.batch_size * N_CELLS
    expected_attr = dataset.batch_size
    val["expected_cell_rows"] = expected_cells
    val["expected_attribution_rows"] = expected_attr
    val["n_cell_rows"] = len(cell_rows)
    val["n_attribution_rows"] = len(attr_rows)
    val["checks"]["row_counts"] = len(cell_rows) == expected_cells and len(attr_rows) == expected_attr
    val["overall_pass"] = all(val["checks"].values())
    (cfg.out_dir / "validation_report.json").write_text(json.dumps(val, indent=2) + "\n")
    run_cfg = {
        "phase": "confirmation",
        "seed": cfg.seed,
        "model": MODEL_ID,
        "model_revision": revision,
        "preregistration_sha": cfg.preregistration_sha,
        "selection_lock": str(cfg.selection_lock_path),
        "primary": lock["primary"],
        "candidate": lock["candidate"],
    }
    (cfg.out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2) + "\n")
    _write_env(cfg.out_dir, cfg.device)
    return val
