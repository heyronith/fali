"""
Extended FALI intervention primitives — score partition, hybrid construction,
pattern/z reconstruction, and cell enumeration.

Model-agnostic tensor conventions (single candidate head):

    scores:  [batch, query, key]
    values:  [batch, key, d_head]
    pattern: [batch, query, key]
    z:       [batch, query, d_head]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Sequence, Tuple

import torch

from extended_fali_math import EXTENDED_FALI_PLAYERS

FactorState = Tuple[int, int, int, int, int]  # (T, R, O, V, C)
CellTable = Dict[FactorState, float]


@dataclass(frozen=True)
class ScorePartition:
    """Boolean masks over [query, key] for T / R / O players."""

    valid_mask: torch.Tensor  # [Q, K]
    target_mask: torch.Tensor  # [Q, K]
    competing_mask: torch.Tensor  # [Q, K]
    other_query_mask: torch.Tensor  # [Q, K]
    final_query: int
    target_key: int


def build_causal_valid_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Structural causal mask: valid[q,k] iff k <= q."""
    q_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    k_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    return k_idx <= q_idx


def build_padding_valid_mask(seq_len: int, valid_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Valid tokens are positions < valid_len."""
    if not (0 <= valid_len <= seq_len):
        raise ValueError(f"valid_len {valid_len} out of range for seq_len {seq_len}")
    pos = torch.arange(seq_len, device=device)
    return pos.unsqueeze(1) < valid_len


def build_score_partition_masks(
    seq_len: int,
    final_query: int,
    target_key: int,
    *,
    valid_len: int | None = None,
    device: torch.device | None = None,
) -> ScorePartition:
    """
    Build structural T/R/O masks for one candidate head score tensor.

    T: (final_query, target_key)
    R: other valid keys in final_query row
    O: valid keys in all other query rows
    """
    if not (0 <= final_query < seq_len and 0 <= target_key < seq_len):
        raise ValueError(f"final_query/target_key out of range for seq_len={seq_len}")
    causal = build_causal_valid_mask(seq_len, device=device)
    if valid_len is None:
        valid = causal.clone()
    else:
        valid = causal & build_padding_valid_mask(seq_len, valid_len, device=device)

    target_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    target_mask[final_query, target_key] = True
    if not bool(valid[final_query, target_key].item()):
        raise ValueError("target coordinate is not structurally valid")

    final_row = valid[final_query].clone()
    competing_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    competing_mask[final_query] = final_row
    competing_mask[final_query, target_key] = False

    other_query_mask = valid.clone()
    other_query_mask[final_query, :] = False

    return ScorePartition(
        valid_mask=valid,
        target_mask=target_mask,
        competing_mask=competing_mask,
        other_query_mask=other_query_mask,
        final_query=final_query,
        target_key=target_key,
    )


def validate_score_partition(partition: ScorePartition) -> None:
    """Require T,R,O partition valid_mask exactly; target structurally valid."""
    t = partition.target_mask
    r = partition.competing_mask
    o = partition.other_query_mask
    v = partition.valid_mask
    if t.shape != v.shape or r.shape != v.shape or o.shape != v.shape:
        raise ValueError("partition masks must share shape")
    if (t & r).any() or (t & o).any() or (r & o).any():
        raise ValueError("T, R, O masks are not pairwise disjoint")
    union = t | r | o
    if not torch.equal(union, v):
        raise ValueError("T ∪ R ∪ O != valid_score_mask")
    if not bool(v[partition.final_query, partition.target_key].item()):
        raise ValueError("target coordinate not in valid mask")


def build_hybrid_scores(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    partition: ScorePartition,
    state: FactorState,
    *,
    mask_fill: float = float("-inf"),
) -> torch.Tensor:
    """
    Mix A/B scores on T/R/O coordinates; force invalid coords to mask_fill.

    scores_*: [batch, query, key]
    state: (T,R,O,V,C) — only first three used here.
    """
    if scores_a.shape != scores_b.shape:
        raise ValueError("scores_a and scores_b shape mismatch")
    if scores_a.ndim != 3:
        raise ValueError(f"expected scores [B,Q,K], got {scores_a.shape}")
    t_on, r_on, o_on, _, _ = state
    out = scores_a.clone()
    for bit, mask in ((t_on, partition.target_mask), (r_on, partition.competing_mask), (o_on, partition.other_query_mask)):
        src = scores_b if bit else scores_a
        out = torch.where(mask.unsqueeze(0), src, out)
    invalid = ~partition.valid_mask
    out = out.masked_fill(invalid.unsqueeze(0), mask_fill)
    return out


def masked_softmax(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    mask_fill: float = float("-inf"),
) -> torch.Tensor:
    """
    Softmax over keys per query on valid coordinates only.

    scores: [batch, query, key]
    valid_mask: [query, key]
    """
    if scores.ndim != 3:
        raise ValueError(f"expected scores [B,Q,K], got {scores.shape}")
    invalid = ~valid_mask
    masked = scores.masked_fill(invalid.unsqueeze(0), mask_fill)
    probs = torch.softmax(masked, dim=-1)
    probs = probs.masked_fill(invalid.unsqueeze(0), 0.0)
    return probs


def reconstruct_z_from_scores_values(
    pattern: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    """
    z[q] = sum_k P[q,k] * V[k]

    pattern: [batch, query, key]
    values:  [batch, key, d_head]
    """
    if pattern.ndim != 3 or values.ndim != 3:
        raise ValueError("pattern must be [B,Q,K] and values [B,K,D]")
    return torch.einsum("bqk,bkd->bqd", pattern, values)


def select_values(values_a: torch.Tensor, values_b: torch.Tensor, v_state: int) -> torch.Tensor:
    return values_b if v_state else values_a


def factor_state_from_mask(mask: int) -> FactorState:
    if mask < 0 or mask >= 32:
        raise ValueError(f"factor mask {mask} out of range")
    return (
        (mask >> 0) & 1,
        (mask >> 1) & 1,
        (mask >> 2) & 1,
        (mask >> 3) & 1,
        (mask >> 4) & 1,
    )


def factor_state_to_players(state: FactorState) -> frozenset[str]:
    names = EXTENDED_FALI_PLAYERS
    return frozenset(names[i] for i, bit in enumerate(state) if bit)


def enumerate_extended_fali_cells() -> Iterator[Tuple[int, FactorState]]:
    """Yield (mask, state) for all 32 binary cells in TR O V C order."""
    for mask in range(32):
        yield mask, factor_state_from_mask(mask)


def cell_table_from_utility_fn(utility_fn) -> CellTable:
    table: CellTable = {}
    for mask, state in enumerate_extended_fali_cells():
        table[state] = float(utility_fn(state))
    return table


def char_fn_from_cell_table(table: CellTable):
    from extended_fali_math import EXTENDED_FALI_PLAYERS, make_char_fn_from_table

    mapping = {factor_state_to_players(state): val for state, val in table.items()}
    return make_char_fn_from_table(EXTENDED_FALI_PLAYERS, mapping)


def quotient_h_values_from_cell_table(table: CellTable) -> Dict[str, float]:
    """
    Extract 8 full-block SCORE × V × C cells for three_factor_shapley.

    SCORE=1 means T=R=O=1; P player corresponds to SCORE block.
    """
    def pick(score: int, v: int, c: int) -> float:
        state: FactorState = (score, score, score, v, c)
        return table[state]

    return {
        "H000": pick(0, 0, 0),
        "H100": pick(1, 0, 0),
        "H010": pick(0, 1, 0),
        "H110": pick(1, 1, 0),
        "H001": pick(0, 0, 1),
        "H101": pick(1, 0, 1),
        "H011": pick(0, 1, 1),
        "H111": pick(1, 1, 1),
    }


def utility_from_injected_head(
    metric_injected: torch.Tensor,
    metric_zero: torch.Tensor,
) -> torch.Tensor:
    """U = M(z_injected) - M(z_zero) per batch element."""
    return metric_injected - metric_zero


def score_pattern_z_pipeline(
    scores_a: torch.Tensor,
    scores_b: torch.Tensor,
    values_a: torch.Tensor,
    values_b: torch.Tensor,
    partition: ScorePartition,
    state: FactorState,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hybrid scores → pattern → z for one cell."""
    hybrid_scores = build_hybrid_scores(scores_a, scores_b, partition, state)
    pattern = masked_softmax(hybrid_scores, partition.valid_mask)
    values = select_values(values_a, values_b, state[3])
    z = reconstruct_z_from_scores_values(pattern, values)
    return hybrid_scores, pattern, z
