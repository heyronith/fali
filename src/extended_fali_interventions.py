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
    """Boolean masks over [batch, query, key] for T / R / O players."""

    valid_mask: torch.Tensor  # [B, Q, K]
    target_mask: torch.Tensor  # [B, Q, K]
    competing_mask: torch.Tensor  # [B, Q, K]
    other_query_mask: torch.Tensor  # [B, Q, K]
    final_query: torch.Tensor  # [B]
    target_key: torch.Tensor  # [B]


def build_causal_valid_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Structural causal mask: valid[q,k] iff k <= q."""
    q_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    k_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    return k_idx <= q_idx


def build_batched_score_partition(
    seq_len: int,
    final_queries: torch.Tensor,
    target_keys: torch.Tensor,
    *,
    valid_lengths: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> ScorePartition:
    """
    Build structural T/R/O masks for a batch of examples.

    For example b, valid[b,q,k] iff q < valid_length[b] AND k <= q.
    Default valid_length[b] = final_queries[b] + 1.
    """
    final_queries = final_queries.to(device=device, dtype=torch.long).reshape(-1)
    target_keys = target_keys.to(device=device, dtype=torch.long).reshape(-1)
    batch = final_queries.numel()
    if target_keys.numel() != batch:
        raise ValueError("final_queries and target_keys must have the same batch size")
    if batch == 0:
        raise ValueError("batch size must be positive")

    if valid_lengths is None:
        valid_lengths = final_queries + 1
    else:
        valid_lengths = valid_lengths.to(device=device, dtype=torch.long).reshape(-1)
        if valid_lengths.numel() != batch:
            raise ValueError("valid_lengths batch size mismatch")

    causal = build_causal_valid_mask(seq_len, device=device)  # [Q,K]
    q_idx = torch.arange(seq_len, device=device)
    valid_mask = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool, device=device)
    target_mask = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool, device=device)
    competing_mask = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool, device=device)
    other_query_mask = torch.zeros(batch, seq_len, seq_len, dtype=torch.bool, device=device)

    for b in range(batch):
        fq = int(final_queries[b].item())
        tk = int(target_keys[b].item())
        vl = int(valid_lengths[b].item())
        if not (0 <= fq < seq_len and 0 <= tk < seq_len):
            raise ValueError(f"example {b}: final_query/target_key out of range for seq_len={seq_len}")
        if not (0 < vl <= seq_len):
            raise ValueError(f"example {b}: valid_length {vl} out of range")

        valid = causal & (q_idx.unsqueeze(1) < vl)
        if not bool(valid[fq, tk].item()):
            raise ValueError(f"example {b}: target coordinate is not structurally valid")

        valid_mask[b] = valid
        target_mask[b, fq, tk] = True

        final_row = valid[fq].clone()
        competing = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        competing[fq] = final_row
        competing[fq, tk] = False
        competing_mask[b] = competing

        other = valid.clone()
        other[fq, :] = False
        other_query_mask[b] = other

    return ScorePartition(
        valid_mask=valid_mask,
        target_mask=target_mask,
        competing_mask=competing_mask,
        other_query_mask=other_query_mask,
        final_query=final_queries,
        target_key=target_keys,
    )


def build_score_partition_masks(
    seq_len: int,
    final_query: int,
    target_key: int,
    *,
    valid_len: int | None = None,
    device: torch.device | None = None,
) -> ScorePartition:
    """Single-example convenience wrapper around build_batched_score_partition."""
    fq = torch.tensor([final_query], device=device, dtype=torch.long)
    tk = torch.tensor([target_key], device=device, dtype=torch.long)
    vl = None if valid_len is None else torch.tensor([valid_len], device=device, dtype=torch.long)
    return build_batched_score_partition(seq_len, fq, tk, valid_lengths=vl, device=device)


def partition_diagnostics(partition: ScorePartition) -> Dict[str, List[int]]:
    """Per-example partition coverage, overlap, and target counts."""
    batch = partition.valid_mask.shape[0]
    coverage_error: List[int] = []
    overlap_count: List[int] = []
    target_count: List[int] = []
    for b in range(batch):
        t = partition.target_mask[b]
        r = partition.competing_mask[b]
        o = partition.other_query_mask[b]
        v = partition.valid_mask[b]
        union = t | r | o
        coverage_error.append(int((union ^ v).sum().item()))
        overlap_count.append(int((t & r).sum().item() + (t & o).sum().item() + (r & o).sum().item()))
        target_count.append(int(t.sum().item()))
    return {
        "coverage_error_per_example": coverage_error,
        "overlap_count_per_example": overlap_count,
        "target_count_per_example": target_count,
    }


def validate_score_partition(partition: ScorePartition) -> None:
    """Require T,R,O partition valid_mask exactly per example; one T coordinate each."""
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

    batch = v.shape[0]
    for b in range(batch):
        fq = int(partition.final_query[b].item())
        tk = int(partition.target_key[b].item())
        if not bool(v[b, fq, tk].item()):
            raise ValueError(f"example {b}: target coordinate not in valid mask")
        if int(t[b].sum().item()) != 1:
            raise ValueError(f"example {b}: T must have exactly one coordinate")


def canonicalize_score_row(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    mode: str = "row_mean",
) -> torch.Tensor:
    """
    Canonicalize attention-score rows within a softmax-equivalence class.

    scores: [batch, query, key]
    valid_mask: [batch, query, key] or broadcastable [query, key]

    mode="native": return scores unchanged (clone not required; returns same tensor).
    mode="row_mean": for each (batch, query) row, subtract the mean over valid keys only.
      Invalid/masked positions do not enter the mean and retain their original values.

    Mean subtraction is computed in float64 for numerical stability, then cast back.
    """
    if mode == "native":
        return scores
    if mode != "row_mean":
        raise ValueError(f"unsupported score gauge mode: {mode!r}")
    if scores.ndim != 3:
        raise ValueError(f"expected scores [B,Q,K], got {scores.shape}")
    if valid_mask.ndim == 2:
        valid_mask = valid_mask.unsqueeze(0).expand(scores.shape[0], -1, -1)
    if valid_mask.shape != scores.shape:
        raise ValueError("valid_mask must match scores shape [B,Q,K] (or [Q,K])")
    if valid_mask.dtype != torch.bool:
        valid_mask = valid_mask.bool()

    n_valid = valid_mask.sum(dim=-1)  # [B, Q]
    if int((n_valid == 0).all().item()) == 1:
        raise ValueError("canonicalize_score_row: no valid keys in any row")
    # Per-example: at least one valid key must exist somewhere.
    per_example = n_valid.sum(dim=-1)  # [B]
    if bool((per_example == 0).any().item()):
        raise ValueError("canonicalize_score_row: example with zero valid keys")

    # Float64 mean over valid keys; rows with zero valid keys leave scores unchanged.
    scores64 = scores.double()
    filled = scores64.masked_fill(~valid_mask, 0.0)
    n = n_valid.clamp(min=1).unsqueeze(-1).to(dtype=torch.float64)
    mu = filled.sum(dim=-1, keepdim=True) / n
    centered = scores64 - mu
    out64 = torch.where(valid_mask, centered, scores64)
    no_valid = (n_valid == 0).unsqueeze(-1)
    out64 = torch.where(no_valid, scores64, out64)
    if not torch.isfinite(out64[valid_mask]).all():
        raise ValueError("canonicalize_score_row produced non-finite valid scores")
    return out64.to(dtype=scores.dtype)


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
    if scores_a.shape[0] != partition.valid_mask.shape[0]:
        raise ValueError("scores batch size must match partition batch size")
    t_on, r_on, o_on, _, _ = state
    out = scores_a.clone()
    for bit, mask in (
        (t_on, partition.target_mask),
        (r_on, partition.competing_mask),
        (o_on, partition.other_query_mask),
    ):
        src = scores_b if bit else scores_a
        out = torch.where(mask, src, out)
    invalid = ~partition.valid_mask
    out = out.masked_fill(invalid, mask_fill)
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
    valid_mask: [batch, query, key] or [query, key] (broadcast over batch)
    """
    if scores.ndim != 3:
        raise ValueError(f"expected scores [B,Q,K], got {scores.shape}")
    if valid_mask.ndim == 2:
        valid_mask = valid_mask.unsqueeze(0).expand(scores.shape[0], -1, -1)
    if valid_mask.shape != scores.shape:
        raise ValueError("valid_mask must be [B,Q,K] or [Q,K] matching scores")
    invalid = ~valid_mask
    masked = scores.masked_fill(invalid, mask_fill)
    probs = torch.softmax(masked, dim=-1)
    probs = probs.masked_fill(invalid, 0.0)
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
    *,
    softmax_dtype: torch.dtype | None = None,
    scores_a_native: torch.Tensor | None = None,
    scores_b_native: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hybrid scores → pattern → z for one cell.

    When scores_*_native are provided and T=R=O (pure endpoint score membership),
    softmax uses the native endpoint scores. That is softmax-equivalent to the
    row-mean gauge on pure-endpoint cells but matches the native FP32 path exactly,
    which is required for endpoint utility invariance under gauge sensitivity.

    Mixed T/R/O cells always use the (possibly gauged) hybrid scores.
    """
    t, r, o, v_bit, _c = state
    use_native_pure = (
        scores_a_native is not None
        and scores_b_native is not None
        and t == r == o
    )
    if use_native_pure:
        src = scores_a_native if t == 0 else scores_b_native
        hybrid_scores = src.masked_fill(~partition.valid_mask, float("-inf"))
    else:
        hybrid_scores = build_hybrid_scores(scores_a, scores_b, partition, state)
    if softmax_dtype is not None and softmax_dtype != hybrid_scores.dtype:
        pattern = masked_softmax(hybrid_scores.to(dtype=softmax_dtype), partition.valid_mask)
        pattern = pattern.to(dtype=scores_a.dtype)
    else:
        pattern = masked_softmax(hybrid_scores, partition.valid_mask)
    values = select_values(values_a, values_b, v_bit)
    z = reconstruct_z_from_scores_values(pattern, values)
    return hybrid_scores, pattern, z
