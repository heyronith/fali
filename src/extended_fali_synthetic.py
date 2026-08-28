"""
EF-1 synthetic validation systems — Level A ground truth and Level B tiny transformer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch

from extended_fali_interventions import (
    CellTable,
    FactorState,
    ScorePartition,
    build_batched_score_partition,
    build_hybrid_scores,
    build_score_partition_masks,
    cell_table_from_utility_fn,
    char_fn_from_cell_table,
    enumerate_extended_fali_cells,
    masked_softmax,
    partition_diagnostics,
    quotient_h_values_from_cell_table,
    reconstruct_z_from_scores_values,
    score_pattern_z_pipeline,
    select_values,
    validate_score_partition,
)
from extended_fali_math import (
    EXTENDED_FALI_GROUPS,
    EXTENDED_FALI_PLAYERS,
    extended_fali_quotient_shapley,
    owen_value_subset,
)
from revision_factorial import three_factor_shapley


@dataclass
class LevelASystem:
    scores_a: torch.Tensor  # [B,Q,K] float64
    scores_b: torch.Tensor
    values_a: torch.Tensor  # [B,K,D]
    values_b: torch.Tensor
    partition: ScorePartition
    baseline_a: float
    baseline_b: float
    readout_a: torch.Tensor  # [Q,D]
    readout_b: torch.Tensor

    def readout(self, context: int) -> torch.Tensor:
        return self.readout_b if context else self.readout_a

    def baseline(self, context: int) -> float:
        return self.baseline_b if context else self.baseline_a

    def metric(self, z: torch.Tensor, context: int) -> torch.Tensor:
        """M_C(z) = baseline_C + sum_q readout_C[q]·z[q] per batch element."""
        readout = self.readout(context)
        contrib = torch.einsum("bqd,qd->b", z, readout)
        return contrib + self.baseline(context)

    def utility(self, state: FactorState) -> float:
        _, pattern, z = score_pattern_z_pipeline(
            self.scores_a,
            self.scores_b,
            self.values_a,
            self.values_b,
            self.partition,
            state,
        )
        c = state[4]
        injected = self.metric(z, c)
        zero = self.metric(torch.zeros_like(z), c)
        return float((injected - zero).mean().item())

    def cell_table(self) -> CellTable:
        return cell_table_from_utility_fn(self.utility)

    def owen(self) -> Dict[str, float]:
        return owen_value_subset(EXTENDED_FALI_PLAYERS, EXTENDED_FALI_GROUPS, char_fn_from_cell_table(self.cell_table()))

    def quotient_bridge_error(self) -> float:
        h = quotient_h_values_from_cell_table(self.cell_table())
        phi_p, phi_v, phi_c, _ = three_factor_shapley(
            h["H000"], h["H100"], h["H010"], h["H110"], h["H001"], h["H101"], h["H011"], h["H111"]
        )
        owen = self.owen()
        q = extended_fali_quotient_shapley(char_fn_from_cell_table(self.cell_table()))
        return max(
            abs(sum(owen[p] for p in ("T", "R", "O")) - phi_p),
            abs(owen["V"] - phi_v),
            abs(owen["C"] - phi_c),
            abs(q["SCORE"] - phi_p),
            abs(q["VALUE"] - phi_v),
            abs(q["CONTEXT"] - phi_c),
        )


def _partition_scalar(partition: ScorePartition) -> Tuple[int, int]:
    return int(partition.final_query[0].item()), int(partition.target_key[0].item())


def _base_system(
    *,
    seq_len: int = 4,
    valid_len: int | None = None,
    d_head: int = 3,
    batch: int = 1,
    final_query: int | None = None,
    target_key: int | None = None,
) -> LevelASystem:
    device = torch.device("cpu")
    dtype = torch.float64
    if final_query is None:
        final_query = seq_len - 1
    if target_key is None:
        target_key = final_query - 1 if final_query >= 1 else 0
    partition = build_score_partition_masks(seq_len, final_query, target_key, valid_len=valid_len, device=device)
    validate_score_partition(partition)

    scores_a = torch.zeros(batch, seq_len, seq_len, dtype=dtype, device=device)
    scores_b = torch.zeros(batch, seq_len, seq_len, dtype=dtype, device=device)
    values_a = torch.zeros(batch, seq_len, d_head, dtype=dtype, device=device)
    values_b = torch.zeros(batch, seq_len, d_head, dtype=dtype, device=device)
    readout_a = torch.zeros(seq_len, d_head, dtype=dtype, device=device)
    readout_b = torch.zeros(seq_len, d_head, dtype=dtype, device=device)
    return LevelASystem(
        scores_a=scores_a,
        scores_b=scores_b,
        values_a=values_a,
        values_b=values_b,
        partition=partition,
        baseline_a=0.0,
        baseline_b=0.0,
        readout_a=readout_a,
        readout_b=readout_b,
    )


def build_level_a_case_target() -> LevelASystem:
    sys = _base_system()
    fq, tk = _partition_scalar(sys.partition)
    sys.scores_a[..., fq, tk] = 0.0
    sys.scores_a[..., fq, :] = -1.0
    sys.scores_a[..., fq, tk] = 1.0
    sys.scores_b = sys.scores_a.clone()
    sys.scores_b[..., fq, tk] = 4.0
    sys.values_a[..., tk, :] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    sys.values_b = sys.values_a.clone()
    sys.readout_a[fq] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    sys.readout_b[fq] = sys.readout_a[fq].clone()
    return sys


def build_level_a_case_competing() -> LevelASystem:
    sys = _base_system()
    fq, tk = _partition_scalar(sys.partition)
    sys.scores_a[..., fq, :] = -1.0
    sys.scores_a[..., fq, tk] = 2.0
    sys.scores_a[..., fq, fq] = 0.5
    sys.scores_b = sys.scores_a.clone()
    sys.scores_b[..., fq, fq] = 3.0
    sys.values_a[..., tk, :] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    sys.values_a[..., fq, :] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    sys.values_b = sys.values_a.clone()
    sys.readout_a[fq] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    sys.readout_b[fq] = sys.readout_a[fq].clone()
    return sys


def build_level_a_case_other_query() -> LevelASystem:
    sys = _base_system(seq_len=4, final_query=3, target_key=2)
    fq, tk = _partition_scalar(sys.partition)
    oq = 1
    sys.scores_a[..., :, :] = -1.0
    sys.scores_a[..., fq, tk] = 1.0
    sys.scores_a[..., oq, oq] = 0.2
    sys.scores_b = sys.scores_a.clone()
    sys.scores_b[..., oq, oq] = 3.0
    sys.values_a[..., tk, :] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    sys.values_a[..., oq, :] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    sys.values_b = sys.values_a.clone()
    sys.readout_a[oq] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    sys.readout_a[fq] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    sys.readout_b = sys.readout_a.clone()
    return sys


def build_level_a_case_value() -> LevelASystem:
    sys = _base_system()
    fq, tk = _partition_scalar(sys.partition)
    sys.scores_a[..., fq, tk] = 2.0
    sys.scores_b = sys.scores_a.clone()
    sys.values_a[..., fq, :] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    sys.values_b[..., fq, :] = torch.tensor([3.0, 0.0, 0.0], dtype=torch.float64)
    sys.readout_a[fq] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    sys.readout_b[fq] = sys.readout_a[fq].clone()
    return sys


def build_level_a_case_context() -> LevelASystem:
    sys = _base_system()
    fq, tk = _partition_scalar(sys.partition)
    sys.scores_a[..., fq, tk] = 2.0
    sys.scores_b = sys.scores_a.clone()
    sys.values_a[..., tk, :] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    sys.values_b = sys.values_a.clone()
    sys.readout_a[fq] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    sys.readout_b[fq] = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
    sys.baseline_a = 0.0
    sys.baseline_b = 0.0
    return sys


def build_level_a_tc_interaction(delta: float = 2.0) -> LevelASystem:
    """
    Pure T×C interaction through score→softmax→z→readout.

    T=0 drives target softmax mass ≈0 (z≈0); T=1 restores mass. C selects readout.
    """
    sys = _base_system()
    fq, tk = _partition_scalar(sys.partition)
    sys.scores_a[..., fq, :] = -20.0
    sys.scores_a[..., fq, tk] = -80.0
    sys.scores_b = sys.scores_a.clone()
    sys.scores_b[..., fq, tk] = 8.0
    sys.values_a[..., tk, :] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    sys.values_b = sys.values_a.clone()
    sys.readout_a[fq] = torch.zeros(3, dtype=torch.float64)
    sys.readout_b[fq] = torch.tensor([delta, 0.0, 0.0], dtype=torch.float64)
    return sys


def endpoint_reconstruction_errors(system: LevelASystem) -> Dict[str, float]:
    part = system.partition
    valid = part.valid_mask
    all_a: FactorState = (0, 0, 0, 0, 0)
    all_b: FactorState = (1, 1, 1, 1, 1)
    hs_a, _, z_a = score_pattern_z_pipeline(
        system.scores_a, system.scores_b, system.values_a, system.values_b, part, all_a
    )
    hs_b, _, z_b = score_pattern_z_pipeline(
        system.scores_a, system.scores_b, system.values_a, system.values_b, part, all_b
    )
    sa_ref = system.scores_a.masked_fill(~valid, float("-inf"))
    sb_ref = system.scores_b.masked_fill(~valid, float("-inf"))
    p_a = masked_softmax(sa_ref, valid)
    p_b = masked_softmax(sb_ref, valid)
    z_a_direct = reconstruct_z_from_scores_values(p_a, system.values_a)
    z_b_direct = reconstruct_z_from_scores_values(p_b, system.values_b)
    table = system.cell_table()
    u0 = table[(0, 0, 0, 0, 0)]
    u1 = table[(1, 1, 1, 1, 1)]
    inj_a = system.metric(z_a, 0) - system.metric(torch.zeros_like(z_a), 0)
    inj_b = system.metric(z_b, 1) - system.metric(torch.zeros_like(z_b), 1)
    intact = float(inj_a.mean())
    cond = float(inj_b.mean())
    gain = cond - intact
    return {
        "scores_a": float(((hs_a - sa_ref).masked_fill(~valid, 0.0)).abs().max()),
        "scores_b": float(((hs_b - sb_ref).masked_fill(~valid, 0.0)).abs().max()),
        "pattern_a": float((masked_softmax(hs_a, valid) - p_a).abs().max()),
        "pattern_b": float((masked_softmax(hs_b, valid) - p_b).abs().max()),
        "z_a": float((z_a - z_a_direct).abs().max()),
        "z_b": float((z_b - z_b_direct).abs().max()),
        "conditional_gain": abs((u1 - u0) - gain),
    }


def build_level_a_batched_partition_system() -> LevelASystem:
    """Three-example batch with distinct final queries, targets, and valid lengths."""
    batch = 3
    seq_len = 6
    d_head = 3
    device = torch.device("cpu")
    dtype = torch.float64
    final_queries = torch.tensor([3, 5, 4], device=device)
    target_keys = torch.tensor([1, 4, 0], device=device)
    valid_lengths = torch.tensor([4, 6, 5], device=device)
    partition = build_batched_score_partition(
        seq_len, final_queries, target_keys, valid_lengths=valid_lengths, device=device
    )
    validate_score_partition(partition)

    scores_a = torch.randn(batch, seq_len, seq_len, dtype=dtype, device=device)
    scores_b = scores_a + 0.5
    values_a = torch.randn(batch, seq_len, d_head, dtype=dtype, device=device)
    values_b = values_a + 0.25
    readout_a = torch.randn(seq_len, d_head, dtype=dtype, device=device)
    readout_b = readout_a.clone()
    return LevelASystem(
        scores_a=scores_a,
        scores_b=scores_b,
        values_a=values_a,
        values_b=values_b,
        partition=partition,
        baseline_a=0.0,
        baseline_b=0.0,
        readout_a=readout_a,
        readout_b=readout_b,
    )


def batched_hybrid_reconstruction_errors(system: LevelASystem) -> Dict[str, float]:
    """Per-batch all-A / all-B hybrid score reconstruction max error."""
    part = system.partition
    valid = part.valid_mask
    batch = valid.shape[0]
    max_a = 0.0
    max_b = 0.0
    for b in range(batch):
        part_b = ScorePartition(
            valid_mask=part.valid_mask[b : b + 1],
            target_mask=part.target_mask[b : b + 1],
            competing_mask=part.competing_mask[b : b + 1],
            other_query_mask=part.other_query_mask[b : b + 1],
            final_query=part.final_query[b : b + 1],
            target_key=part.target_key[b : b + 1],
        )
        sa = system.scores_a[b : b + 1]
        sb = system.scores_b[b : b + 1]
        hs_a = build_hybrid_scores(sa, sb, part_b, (0, 0, 0, 0, 0))
        hs_b = build_hybrid_scores(sa, sb, part_b, (1, 1, 1, 0, 0))
        vb = valid[b : b + 1]
        sa_m = sa.masked_fill(~vb, float("-inf"))
        sb_m = sb.masked_fill(~vb, float("-inf"))
        max_a = max(max_a, float(((hs_a - sa_m).masked_fill(~vb, 0.0)).abs().max()))
        max_b = max(max_b, float(((hs_b - sb_m).masked_fill(~vb, 0.0)).abs().max()))
    return {"all_a_max": max_a, "all_b_max": max_b}


def single_channel_owen_check(system: LevelASystem, intended: str) -> Dict[str, float]:
    owen = system.owen()
    table = system.cell_table()
    delta = table[(1, 1, 1, 1, 1)] - table[(0, 0, 0, 0, 0)]
    out = {"delta_u": delta, "intended": owen[intended], "intended_err": abs(owen[intended] - delta)}
    for p in EXTENDED_FALI_PLAYERS:
        if p != intended:
            out[f"other_{p}"] = abs(owen[p])
    return out


LEVEL_A_CASES: Dict[str, Tuple[Callable[[], LevelASystem], str]] = {
    "T": (build_level_a_case_target, "T"),
    "R": (build_level_a_case_competing, "R"),
    "O": (build_level_a_case_other_query, "O"),
    "V": (build_level_a_case_value, "V"),
    "C": (build_level_a_case_context, "C"),
}
