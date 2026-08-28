"""
Extended FALI on GPT-2 IOI — partition, validation, and intervention helpers.

Used by EF-2 smoke validation only; not scientific inference code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch

from extended_fali_interventions import (
    CellTable,
    FactorState,
    ScorePartition,
    build_batched_score_partition,
    char_fn_from_cell_table,
    enumerate_extended_fali_cells,
    factor_state_to_players,
    masked_softmax,
    partition_diagnostics,
    quotient_h_values_from_cell_table,
    reconstruct_z_from_scores_values,
    score_pattern_z_pipeline,
    validate_score_partition,
)
from extended_fali_math import (
    EXTENDED_FALI_GROUPS,
    EXTENDED_FALI_PLAYERS,
    extended_fali_quotient_shapley,
    make_char_fn_from_table,
    owen_value_subset,
    verify_efficiency,
    verify_owen_quotient_shapley_consistency,
)
from revision_config import Head, PRIMARY_NAME_MOVERS
from revision_factorial import build_factorial_fwd_hooks, max_abs_error, three_factor_shapley
from revision_fali import extract_target_and_competing, fali_shapley
from revision_interventions import build_zero_ablation_hooks

# EF-2 frozen validation thresholds (GPT-2 smoke).
SCORE_PATTERN_TOL = 2e-6
Z_RECON_TOL = 1e-5
ENDPOINT_TOL = 1e-4
OWEN_TOL = 1e-8
QUOTIENT_TOL = 1e-8
HISTORICAL_TOL = 1e-4
SAME_LAYER_SCORE_TOL = 1e-6
SAME_LAYER_OWEN_TOL = 1e-6
INJECTION_SCOPE_TOL = 1e-7
LEAK_TOL = 1e-5
POST_END_TOL = 1e-6

HISTORICAL_H_COLS = [
    "H000",
    "H100",
    "H010",
    "H110",
    "H001",
    "H101",
    "H011",
    "H111",
]
HISTORICAL_PHI_COLS = ["phi_pattern", "phi_value", "phi_context"]

SAME_LAYER_HEADS = ("L9H7", "L9H0")
INJECTION_SCOPE_HEADS = ("L10H2", "L9H7")

QUOTIENT_STATE_MAP = {
    "H000": (0, 0, 0, 0, 0),
    "H100": (1, 1, 1, 0, 0),
    "H010": (0, 0, 0, 1, 0),
    "H110": (1, 1, 1, 1, 0),
    "H001": (0, 0, 0, 0, 1),
    "H101": (1, 1, 1, 0, 1),
    "H011": (0, 0, 0, 1, 1),
    "H111": (1, 1, 1, 1, 1),
}


def build_ioi_partition(dataset, device: torch.device | str) -> ScorePartition:
    """Task-derived T/R/O partition: end query, IO key, valid_length = end+1."""
    end = dataset.end.to(device=device, dtype=torch.long)
    target = dataset.IO.to(device=device, dtype=torch.long)
    valid_length = end + 1
    seq_len = int(dataset.clean_tokens.shape[1])
    partition = build_batched_score_partition(
        seq_len, end, target, valid_lengths=valid_length, device=device
    )
    validate_score_partition(partition)
    return partition


def compute_ios(logits: torch.Tensor, dataset) -> torch.Tensor:
    """Per-example IO−S at dataset.end."""
    batch = dataset.batch_size
    device = logits.device
    end_idx = dataset.end.to(device)
    batch_idx = torch.arange(batch, device=device)
    logits_at_end = logits[batch_idx, end_idx]
    io = logits_at_end[batch_idx, dataset.io_token_ids.to(device)]
    s = logits_at_end[batch_idx, dataset.s_token_ids.to(device)]
    return io - s


def max_valid_coord_error(a: torch.Tensor, b: torch.Tensor, valid_mask: torch.Tensor) -> float:
    diff = (a - b).abs().masked_fill(~valid_mask, 0.0)
    return float(diff.max().item())


def max_z_error_on_prompt(
    z_rec: torch.Tensor,
    z_cached: torch.Tensor,
    end_positions: torch.Tensor,
) -> float:
    """Max |z_rec - z_cached| over q <= end[b] (post-end rows excluded)."""
    max_err = 0.0
    for b in range(z_rec.shape[0]):
        end = int(end_positions[b].item())
        err = (z_rec[b, : end + 1] - z_cached[b, : end + 1]).abs().max()
        max_err = max(max_err, float(err.item()))
    return max_err


def cache_names_filter(layers: Sequence[int]) -> List[str]:
    names: List[str] = []
    for layer in layers:
        names.extend(
            [
                f"blocks.{layer}.attn.hook_attn_scores",
                f"blocks.{layer}.attn.hook_pattern",
                f"blocks.{layer}.attn.hook_v",
                f"blocks.{layer}.attn.hook_z",
            ]
        )
    return names


def extract_head_cache(
    cache,
    layer: int,
    head: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = cache[f"blocks.{layer}.attn.hook_attn_scores"][:, head].contiguous()
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"][:, head].contiguous()
    values = cache[f"blocks.{layer}.attn.hook_v"][:, :, head, :].contiguous()
    z = cache[f"blocks.{layer}.attn.hook_z"][:, :, head, :].contiguous()
    return scores, pattern, values, z


def validate_score_pattern(
    scores: torch.Tensor,
    pattern_cached: torch.Tensor,
    partition: ScorePartition,
) -> Tuple[float, float]:
    valid = partition.valid_mask
    p_rec = masked_softmax(scores.masked_fill(~valid, float("-inf")), valid)
    err = max_valid_coord_error(p_rec, pattern_cached, valid)
    invalid = ~valid
    masked_max = float(p_rec.masked_select(invalid).max().item()) if invalid.any() else 0.0
    return err, masked_max


def validate_z_reconstruction(
    pattern: torch.Tensor,
    values: torch.Tensor,
    z_cached: torch.Tensor,
    end_positions: torch.Tensor,
) -> float:
    z_rec = reconstruct_z_from_scores_values(pattern, values)
    return max_z_error_on_prompt(z_rec, z_cached, end_positions)


def owen_from_cell_table(table: CellTable) -> Dict[str, float]:
    return owen_value_subset(
        EXTENDED_FALI_PLAYERS, EXTENDED_FALI_GROUPS, char_fn_from_cell_table(table)
    )


def owen_diagnostics(table: CellTable) -> Dict[str, float]:
    owen = owen_from_cell_table(table)
    char_fn = char_fn_from_cell_table(table)
    delta_u = table[(1, 1, 1, 1, 1)] - table[(0, 0, 0, 0, 0)]
    residual = verify_efficiency(EXTENDED_FALI_PLAYERS, char_fn, owen, atol=OWEN_TOL)
    block_res = verify_owen_quotient_shapley_consistency(
        EXTENDED_FALI_PLAYERS, EXTENDED_FALI_GROUPS, char_fn, owen
    )
    phi_score = owen["T"] + owen["R"] + owen["O"]
    score_res = float(block_res.get("block_0_G0", 0.0))
    value_res = float(block_res.get("block_1_G1", 0.0))
    context_res = float(block_res.get("block_2_G2", 0.0))
    max_block = max(score_res, value_res, context_res)
    return {
        "owen_residual": residual,
        "phi_T": owen["T"],
        "phi_R": owen["R"],
        "phi_O": owen["O"],
        "phi_V": owen["V"],
        "phi_C": owen["C"],
        "Phi_SCORE": phi_score,
        "quotient_score_residual": score_res,
        "quotient_value_residual": value_res,
        "quotient_context_residual": context_res,
        "quotient_block_max": max_block,
        "Delta_U": delta_u,
    }


def file_sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_paths(repo: Path, relative_paths: Sequence[str]) -> Dict[str, Dict]:
    snap: Dict[str, Dict] = {}
    for rel in relative_paths:
        p = repo / rel
        st = p.stat() if p.exists() else None
        snap[rel] = {"sha256": file_sha256(p), "size": st.st_size if st else None}
    return snap


def verify_historical_hash_integrity(
    before: Dict[str, Dict],
    after: Dict[str, Dict],
    required_paths: Sequence[str],
) -> Tuple[bool, Dict[str, str]]:
    """
    Verify required historical files exist and are unchanged.

    Missing files => NOT_AVAILABLE and validation failure.
    Never treat sha256=null before/after as integrity evidence.
    """
    checks: Dict[str, str] = {}
    ok = True
    for rel in required_paths:
        b_hash = before.get(rel, {}).get("sha256")
        a_hash = after.get(rel, {}).get("sha256")
        if b_hash is None or a_hash is None:
            checks[rel] = "NOT_AVAILABLE"
            ok = False
        elif b_hash != a_hash:
            checks[rel] = "CHANGED"
            ok = False
        else:
            checks[rel] = "UNCHANGED"
    return ok, checks


def historical_quotient_cells(table: CellTable) -> Dict[str, float]:
    return quotient_h_values_from_cell_table(table)


def compare_historical_cells(
    new_h: Dict[str, float],
    hist_row: pd.Series,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for col in HISTORICAL_H_COLS:
        out[col] = abs(float(new_h[col]) - float(hist_row[col]))
    return out


def compare_historical_shapley(
    owen: Dict[str, float],
    hist_row: pd.Series,
) -> Dict[str, float]:
    phi_score = owen["T"] + owen["R"] + owen["O"]
    return {
        "phi_pattern": abs(phi_score - float(hist_row["phi_pattern"])),
        "phi_value": abs(owen["V"] - float(hist_row["phi_value"])),
        "phi_context": abs(owen["C"] - float(hist_row["phi_context"])),
    }


def descriptive_fali_diag(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    final_query: int,
    target_key: int,
) -> Dict[str, float]:
    t_a, b_a, _ = extract_target_and_competing(scores_a[final_query], target_key, final_query)
    t_b, b_b, _ = extract_target_and_competing(scores_b[final_query], target_key, final_query)
    res = fali_shapley(float(t_a.item()), b_a, float(t_b.item()), b_b)
    return {
        "desc_phi_target": res.phi_target,
        "desc_phi_competing": res.phi_competing,
        "desc_total_delta_attn": res.total_delta_attn,
    }


def align_dataset_to_reference(
    meta: pd.DataFrame,
    reference_csv: pd.DataFrame,
    *,
    token_tensor: Optional[torch.Tensor] = None,
) -> Tuple[str, List[str]]:
    """
    Compare generated metadata to frozen reference. Returns (status, detail_lines).
    """
    cols = [
        "example_idx",
        "prompt",
        "S",
        "IO",
        "template_id",
        "end_position",
        "IO_position",
        "S1_position",
        "S2_position",
    ]
    ref = reference_csv.head(len(meta)).reset_index(drop=True)
    gen = meta.reset_index(drop=True)
    mismatches: List[str] = []
    for c in cols:
        for i in range(len(gen)):
            a = str(gen.loc[i, c])
            b = str(ref.loc[i, c])
            if a != b:
                mismatches.append(f"row {i} col {c}: generated={a!r} ref={b!r}")
    lines = [f"Compared {len(gen)} examples against reference."]
    if mismatches:
        lines.extend(mismatches[:20])
        if len(mismatches) > 20:
            lines.append(f"... and {len(mismatches) - 20} more")
        return "FAIL", lines
    lines.append("PASS: all metadata columns match.")
    if token_tensor is not None:
        h = hashlib.sha256(token_tensor.detach().cpu().numpy().tobytes()).hexdigest()
        lines.append(f"token_tensor_sha256={h}")
    return "PASS", lines


def align_canonical_dataset_to_reference(
    meta: pd.DataFrame,
    reference_csv: pd.DataFrame,
    *,
    token_tensor: Optional[torch.Tensor] = None,
) -> Tuple[str, List[str]]:
    """Compare canonical StandardIOI metadata to frozen Phase-4 reference."""
    cols = [
        "example_idx",
        "low_level_template_id",
        "pattern_type",
        "template_stratum",
        "prompt",
        "S",
        "IO",
        "S1_position",
        "S2_position",
        "IO_position",
        "end_position",
        "place",
        "object",
    ]
    ref = reference_csv.head(len(meta)).reset_index(drop=True)
    gen = meta.reset_index(drop=True)
    mismatches: List[str] = []
    for c in cols:
        if c not in gen.columns or c not in ref.columns:
            mismatches.append(f"missing column {c}")
            continue
        for i in range(len(gen)):
            a = str(gen.loc[i, c])
            b = str(ref.loc[i, c])
            if a != b:
                mismatches.append(f"row {i} col {c}: generated={a!r} ref={b!r}")
    lines = [f"Compared {len(gen)} canonical examples against Phase-4 reference."]
    if mismatches:
        lines.extend(mismatches[:20])
        if len(mismatches) > 20:
            lines.append(f"... and {len(mismatches) - 20} more")
        return "FAIL", lines
    lines.append("PASS: all metadata columns match.")
    if token_tensor is not None:
        h = hashlib.sha256(token_tensor.detach().cpu().numpy().tobytes()).hexdigest()
        lines.append(f"token_tensor_sha256={h}")
    return "PASS", lines


def run_injection_scope_check(
    model,
    tokens: torch.Tensor,
    dataset,
    candidate: Head,
    z_hybrid: torch.Tensor,
    *,
    primary_ablate: bool,
) -> Dict[str, float]:
    layer, head = candidate
    captured: Dict[str, torch.Tensor] = {}
    hooks: List = []
    if primary_ablate:
        hooks.extend(build_zero_ablation_hooks(PRIMARY_NAME_MOVERS))

    def hook_fn(z: torch.Tensor, hook) -> torch.Tensor:
        captured["before"] = z.clone()
        out = z.clone()
        out[:, :, head, :] = z_hybrid
        captured["after"] = out.clone()
        return out

    hooks.append((f"blocks.{layer}.attn.hook_z", hook_fn))
    with torch.no_grad():
        with model.hooks(fwd_hooks=hooks):
            model(tokens)
    before = captured["before"]
    after = captured["after"]
    noncandidate = 0.0
    for h in range(model.cfg.n_heads):
        if h == head:
            continue
        err = float((before[:, :, h, :] - after[:, :, h, :]).abs().max().item())
        noncandidate = max(noncandidate, err)
    candidate_err = float((after[:, :, head, :] - z_hybrid).abs().max().item())
    return {"candidate_z_error": candidate_err, "noncandidate_z_error": noncandidate}


def run_live_leak_check(
    model,
    tokens: torch.Tensor,
    dataset,
    candidate: Head,
    z_hybrid: torch.Tensor,
    *,
    primary_ablate: bool,
    perturb: str,
) -> float:
    layer, head = candidate
    mag = 1e6

    def build_hooks(include_perturb: bool) -> List:
        hooks: List = []
        if primary_ablate:
            hooks.extend(build_zero_ablation_hooks(PRIMARY_NAME_MOVERS))
        if include_perturb:
            if perturb in ("v", "qv"):

                def v_hook(v: torch.Tensor, hook) -> torch.Tensor:
                    out = v.clone()
                    out[:, :, head, :] = out[:, :, head, :] + mag
                    return out

                hooks.insert(0, (f"blocks.{layer}.attn.hook_v", v_hook))
            if perturb in ("q", "qv"):

                def q_hook(q: torch.Tensor, hook) -> torch.Tensor:
                    out = q.clone()
                    out[:, :, head, :] = out[:, :, head, :] + mag
                    return out

                hooks.insert(0, (f"blocks.{layer}.attn.hook_q", q_hook))

        def z_hook(z: torch.Tensor, hook) -> torch.Tensor:
            out = z.clone()
            out[:, :, head, :] = z_hybrid
            return out

        hooks.append((f"blocks.{layer}.attn.hook_z", z_hook))
        return hooks

    def ios(hooks) -> torch.Tensor:
        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                logits = model(tokens)
        return compute_ios(logits, dataset)

    base = ios(build_hooks(include_perturb=False))
    pert = ios(build_hooks(include_perturb=True))
    return float((base - pert).abs().max().item())


def run_post_end_causality_check(
    model,
    tokens: torch.Tensor,
    dataset,
    candidate: Head,
    z_hybrid: torch.Tensor,
) -> float:
    z_pert = z_hybrid.clone()
    for b in range(z_pert.shape[0]):
        end = int(dataset.end[b].item())
        if end + 1 < z_pert.shape[1]:
            z_pert[b, end + 1 :, :] += 1e6
    hooks = build_factorial_fwd_hooks(candidate, context_B=False, z_head=z_hybrid, zero_candidate=False)
    hooks_pert = build_factorial_fwd_hooks(candidate, context_B=False, z_head=z_pert, zero_candidate=False)

    def ios(hk):
        with torch.no_grad():
            with model.hooks(fwd_hooks=hk):
                logits = model(tokens)
        return compute_ios(logits, dataset)

    base = ios(hooks)
    pert = ios(hooks_pert)
    return float((base - pert).abs().max().item())
