"""
Revision FALI: descriptive attention-space Shapley decomposition.

Terminology (mandatory):
  - target-score contribution
  - competing-score contribution

Do NOT call these active routing / passive effect / causal recruitment.
FALI here is initially a descriptive attention-space decomposition under the
corrected primary Name-Mover knockout (L9H6 + L9H9 + L10H0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def causal_key_mask(seq_len: int, query_pos: int, device=None, dtype=torch.bool) -> torch.Tensor:
    """True for causally valid key positions k <= query_pos."""
    return torch.arange(seq_len, device=device) <= int(query_pos)


def is_masked_score(score: torch.Tensor) -> torch.Tensor:
    """Detect -inf / nan masked attention scores."""
    return ~torch.isfinite(score)


def softmax_target_from_scores(
    target_score: torch.Tensor,
    competing_scores: torch.Tensor,
) -> torch.Tensor:
    """
    Softmax probability of the target among {target} ∪ competing.

    target_score: scalar or [batch]
    competing_scores: [n_comp] or [batch, n_comp]
    """
    if target_score.ndim == 0:
        logits = torch.cat([target_score.unsqueeze(0), competing_scores], dim=0)
        return torch.softmax(logits, dim=0)[0]
    # [batch]
    t = target_score.unsqueeze(-1)  # [B, 1]
    logits = torch.cat([t, competing_scores], dim=-1)
    return torch.softmax(logits, dim=-1)[..., 0]


def competing_logsumexp(competing_scores: torch.Tensor) -> torch.Tensor:
    """logsumexp over competing scores. Empty competitors -> -inf."""
    if competing_scores.numel() == 0:
        return torch.tensor(float("-inf"), device=competing_scores.device, dtype=competing_scores.dtype)
    if competing_scores.ndim == 1:
        return torch.logsumexp(competing_scores, dim=0)
    return torch.logsumexp(competing_scores, dim=-1)


def attention_margin(target_score: torch.Tensor, competing_scores: torch.Tensor) -> torch.Tensor:
    """margin = target_score - logsumexp(competing)."""
    return target_score - competing_logsumexp(competing_scores)


def sigmoid_margin_prob(margin: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(margin)


@dataclass
class FALIResult:
    p00: float
    p10: float
    p01: float
    p11: float
    phi_target: float
    phi_competing: float
    residual: float
    total_delta_attn: float


def fali_shapley(
    t_A: float,
    b_A: torch.Tensor,
    t_B: float,
    b_B: torch.Tensor,
) -> FALIResult:
    """
    Target-score and competing-score Shapley of attention probability change.

    p00 = softmax_target(t_A, b_A)
    p10 = softmax_target(t_B, b_A)
    p01 = softmax_target(t_A, b_B)
    p11 = softmax_target(t_B, b_B)

    phi_target = 0.5 * [(p10 - p00) + (p11 - p01)]
    phi_competing = 0.5 * [(p01 - p00) + (p11 - p10)]
    """
    tA = torch.as_tensor(t_A, dtype=torch.float64)
    tB = torch.as_tensor(t_B, dtype=torch.float64)
    # Always compute on CPU float64 for stable descriptive algebra
    bA = b_A.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    bB = b_B.detach().to(device="cpu", dtype=torch.float64).reshape(-1)

    p00 = float(softmax_target_from_scores(tA, bA).item())
    p10 = float(softmax_target_from_scores(tB, bA).item())
    p01 = float(softmax_target_from_scores(tA, bB).item())
    p11 = float(softmax_target_from_scores(tB, bB).item())

    phi_target = 0.5 * ((p10 - p00) + (p11 - p01))
    phi_competing = 0.5 * ((p01 - p00) + (p11 - p10))
    total = p11 - p00
    residual = (phi_target + phi_competing) - total
    return FALIResult(
        p00=p00,
        p10=p10,
        p01=p01,
        p11=p11,
        phi_target=phi_target,
        phi_competing=phi_competing,
        residual=residual,
        total_delta_attn=total,
    )


def extract_target_and_competing(
    score_row: torch.Tensor,
    target_key: int,
    query_pos: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    From a full pre-softmax score row [seq], extract target scalar and competing
    causally-valid finite scores (excluding target).

    Returns: (target_score, competing_scores, valid_mask_bool excluding only target for competing)
    """
    row = score_row.detach()
    seq_len = row.shape[-1]
    if query_pos is None:
        # Infer: last finite position is a common proxy; prefer explicit query_pos
        finite = torch.isfinite(row)
        assert finite.any(), "Score row has no finite entries"
        query_pos = int(torch.where(finite)[0].max().item())

    valid = causal_key_mask(seq_len, query_pos, device=row.device)
    # Also require finite (preserve -inf mask exactly by excluding them)
    finite = torch.isfinite(row)
    valid = valid & finite

    assert 0 <= int(target_key) < seq_len, f"target_key {target_key} out of range"
    assert valid[int(target_key)], (
        f"Target key {target_key} is not causally valid / is masked at query={query_pos}"
    )

    t = row[int(target_key)]
    competing_idx = valid.clone()
    competing_idx[int(target_key)] = False
    competing = row[competing_idx]
    return t, competing, valid


def qk_score(q: torch.Tensor, k: torch.Tensor, d_head: int) -> torch.Tensor:
    """Attention score = (q · k) / sqrt(d_head)."""
    return (q * k).sum(dim=-1) / (float(d_head) ** 0.5)


def qk_target_shapley(
    q_A: torch.Tensor,
    q_B: torch.Tensor,
    k_A: torch.Tensor,
    k_B: torch.Tensor,
    d_head: int,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Descriptive geometric Q/K Shapley on the target IO key score only.

    Returns:
      s_AA, s_BA, s_AB, s_BB, phi_query, phi_key, residual
    """
    s_AA = float(qk_score(q_A, k_A, d_head).item())
    s_BA = float(qk_score(q_B, k_A, d_head).item())
    s_AB = float(qk_score(q_A, k_B, d_head).item())
    s_BB = float(qk_score(q_B, k_B, d_head).item())
    phi_q = 0.5 * ((s_BA - s_AA) + (s_BB - s_AB))
    phi_k = 0.5 * ((s_AB - s_AA) + (s_BB - s_BA))
    residual = (phi_q + phi_k) - (s_BB - s_AA)
    return s_AA, s_BA, s_AB, s_BB, phi_q, phi_k, residual


def build_hybrid_score_row(
    scores_A_row: torch.Tensor,
    scores_B_row: torch.Tensor,
    target_key: int,
    query_pos: int,
    mode: str,
) -> torch.Tensor:
    """
    Build a hybrid score row for modes F00/F10/F01/F11.

    F00: (t_A, b_A) — entire valid row from A
    F10: (t_B, b_A)
    F01: (t_A, b_B)
    F11: (t_B, b_B) — entire valid row from B

    Masked / future positions stay exactly as in the B row (caller should use B
    as the live tensor base); this function returns a full replacement row that
    preserves -inf where either A or B is masked on valid positions, and copies
    B's masked future entries.
    """
    assert mode in {"F00", "F10", "F01", "F11"}
    a = scores_A_row.detach()
    b = scores_B_row.detach()
    assert a.shape == b.shape

    out = b.clone()
    valid = causal_key_mask(a.shape[-1], query_pos, device=a.device)

    # Preserve exact -inf mask on future positions (already from B).
    # On causally valid positions, construct hybrid from A/B.
    if mode == "F00":
        out[valid] = a[valid]
    elif mode == "F11":
        out[valid] = b[valid]
    elif mode == "F10":
        out[valid] = a[valid]
        out[int(target_key)] = b[int(target_key)]
    elif mode == "F01":
        out[valid] = b[valid]
        out[int(target_key)] = a[int(target_key)]

    # Exact mask preservation: wherever B is non-finite, keep B's value
    nonfinite_b = ~torch.isfinite(b)
    out = torch.where(nonfinite_b, b, out)
    # And wherever A is non-finite on positions we took from A, keep -inf
    if mode in {"F00", "F10"}:
        from_a = valid.clone()
        if mode == "F10":
            from_a[int(target_key)] = False
        nonfinite_a = (~torch.isfinite(a)) & from_a
        out = torch.where(nonfinite_a, a, out)
    if mode == "F01":
        if not torch.isfinite(a[int(target_key)]):
            out[int(target_key)] = a[int(target_key)]

    return out
