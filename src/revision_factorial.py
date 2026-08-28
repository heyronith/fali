"""
Phase-3C: full-head P × V × C factorial decomposition of conditional_gain.

Factors (binary):
  P — attention pattern (A vs B)
  V — value vectors (A vs B)
  C — surrounding model context (clean A vs primary-knockout B)

Head effect:
  H(P,V,C) = IO−S(context C with z(P,V)) − IO−S(context C with z=0)

Interventions act on the candidate head's FULL hook_z[:, :, head, :]
(all query positions), matching Phase-1 ablation semantics.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from revision_config import PRIMARY_NAME_MOVERS, Head
from revision_interventions import HookSpec, group_heads_by_layer
from revision_score_interventions import merge_hook_lists


def reconstruct_z_head(
    pattern: torch.Tensor,
    v: torch.Tensor,
    head: int,
) -> torch.Tensor:
    """
    Reconstruct one head's z from pattern and V.

    pattern: [batch, n_heads, q, k]
    v:       [batch, seq, n_heads, d_head]
    returns: [batch, seq, d_head]  (seq == q == k length)
    """
    assert pattern.ndim == 4 and v.ndim == 4
    P = pattern[:, head]  # [B, Q, K]
    V = v[:, :, head, :]  # [B, K, d]
    # z[q] = sum_k P[q,k] * V[k]
    return torch.einsum("bqk,bkd->bqd", P, V)


def reconstruct_all_hybrid_z(
    pattern_A: torch.Tensor,
    pattern_B: torch.Tensor,
    v_A: torch.Tensor,
    v_B: torch.Tensor,
    head: int,
) -> Dict[str, torch.Tensor]:
    """Build z_00, z_10, z_01, z_11 for one head."""
    return {
        "z_00": reconstruct_z_head(pattern_A, v_A, head),
        "z_10": reconstruct_z_head(pattern_B, v_A, head),
        "z_01": reconstruct_z_head(pattern_A, v_B, head),
        "z_11": reconstruct_z_head(pattern_B, v_B, head),
    }


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)).item())


def three_factor_shapley(
    H000: float,
    H100: float,
    H010: float,
    H110: float,
    H001: float,
    H101: float,
    H011: float,
    H111: float,
) -> Tuple[float, float, float, float]:
    """
    Exact Shapley attributions for f:{0,1}^3 → R along P,V,C.

    Returns (phi_pattern, phi_value, phi_context, residual)
    where residual = (phi_P + phi_V + phi_C) - (H111 - H000).
    """
    # Weights: empty/size-2 coalitions → 1/3; size-1 → 1/6
    phi_P = (
        (1.0 / 3.0) * (H100 - H000)
        + (1.0 / 6.0) * (H110 - H010)
        + (1.0 / 6.0) * (H101 - H001)
        + (1.0 / 3.0) * (H111 - H011)
    )
    phi_V = (
        (1.0 / 3.0) * (H010 - H000)
        + (1.0 / 6.0) * (H110 - H100)
        + (1.0 / 6.0) * (H011 - H001)
        + (1.0 / 3.0) * (H111 - H101)
    )
    phi_C = (
        (1.0 / 3.0) * (H001 - H000)
        + (1.0 / 6.0) * (H101 - H100)
        + (1.0 / 6.0) * (H011 - H010)
        + (1.0 / 3.0) * (H111 - H110)
    )
    total = H111 - H000
    residual = (phi_P + phi_V + phi_C) - total
    return phi_P, phi_V, phi_C, residual


def factorial_interactions(
    H000: float,
    H100: float,
    H010: float,
    H110: float,
    H001: float,
    H101: float,
    H011: float,
    H111: float,
) -> Dict[str, float]:
    """
    Explicit ANOVA-style main effects and interactions for the 2×2×2.

    Identity:
      H111 - H000 = main_P + main_V + main_C
                    + interaction_PV + interaction_PC + interaction_VC
                    + interaction_PVC
    """
    main_P = H100 - H000
    main_V = H010 - H000
    main_C = H001 - H000
    interaction_PV = H110 - H100 - H010 + H000
    interaction_PC = H101 - H100 - H001 + H000
    interaction_VC = H011 - H010 - H001 + H000
    interaction_PVC = (
        H111 - H110 - H101 - H011 + H100 + H010 + H001 - H000
    )
    total = H111 - H000
    recon = (
        main_P
        + main_V
        + main_C
        + interaction_PV
        + interaction_PC
        + interaction_VC
        + interaction_PVC
    )
    return {
        "main_P": main_P,
        "main_V": main_V,
        "main_C": main_C,
        "interaction_PV": interaction_PV,
        "interaction_PC": interaction_PC,
        "interaction_VC": interaction_VC,
        "interaction_PVC": interaction_PVC,
        "interaction_recon_residual": recon - total,
    }


def endpoint_marginal_effects(
    H000: float,
    H100: float,
    H010: float,
    H110: float,
    H001: float,
    H101: float,
    H011: float,
    H111: float,
) -> Dict[str, float]:
    """Marginal factor flips at clean and knockout endpoints (not Shapley)."""
    return {
        "pattern_effect_B_VB": H111 - H011,  # flip P at C=1,V=1
        "value_effect_B_PB": H111 - H101,  # flip V at C=1,P=1
        "context_effect_PB_VB": H111 - H110,  # flip C at P=1,V=1
        "pattern_effect_A_VA": H100 - H000,  # flip P at C=0,V=0
        "value_effect_A_PA": H010 - H000,  # flip V at C=0,P=0
        "context_effect_PA_VA": H001 - H000,  # flip C at P=0,V=0
    }


def make_full_z_injection_hook(
    layer: int,
    head: int,
    z_head: Optional[torch.Tensor],
    *,
    zero_heads: Sequence[int] = (),
    zero_candidate: bool = False,
) -> HookSpec:
    """
    Clone-safe hook_z intervention at one layer.

    - Always zeros heads listed in zero_heads (e.g. other primaries on this layer).
    - If zero_candidate: zeros `head`.
    - Else if z_head is not None: replaces [:, :, head, :] with z_head [B, seq, d].
    """
    hook_name = f"blocks.{layer}.attn.hook_z"
    zero_set = sorted(set(int(h) for h in zero_heads))
    z_head_det = None if z_head is None else z_head.detach()

    def hook_fn(z: torch.Tensor, hook) -> torch.Tensor:
        assert z.ndim == 4
        out = z.clone()
        for h in zero_set:
            out[:, :, h, :] = 0.0
        if zero_candidate:
            out[:, :, head, :] = 0.0
        elif z_head_det is not None:
            zh = z_head_det.to(device=out.device, dtype=out.dtype)
            assert zh.shape == out[:, :, head, :].shape, (
                f"z_head shape {tuple(zh.shape)} != "
                f"{tuple(out[:, :, head, :].shape)}"
            )
            out[:, :, head, :] = zh
        return out

    return hook_name, hook_fn


def build_factorial_fwd_hooks(
    candidate: Head,
    *,
    context_B: bool,
    z_head: Optional[torch.Tensor],
    zero_candidate: bool,
) -> List[HookSpec]:
    """
    Build composed hooks for one factorial cell.

    context_B=True → apply canonical primary Name-Mover knockout.
    Candidate head receives either z_head replacement or zeroing.
    """
    layer, head = int(candidate[0]), int(candidate[1])
    primaries = list(PRIMARY_NAME_MOVERS)

    # Group primary zeros by layer; exclude candidate from primary list if present
    primary_by_layer = group_heads_by_layer(
        [(l, h) for (l, h) in primaries if not (l == layer and h == head)]
    )

    hooks: List[HookSpec] = []

    if context_B:
        for L, heads in primary_by_layer.items():
            if L == layer:
                # Same layer: fold primary zeros into the candidate injection hook
                hooks.append(
                    make_full_z_injection_hook(
                        layer,
                        head,
                        z_head,
                        zero_heads=heads,
                        zero_candidate=zero_candidate,
                    )
                )
            else:
                # Other layers: pure primary zero
                hooks.append(
                    make_full_z_injection_hook(
                        L,
                        heads[0],  # unused if zero_candidate False and z None
                        None,
                        zero_heads=heads,
                        zero_candidate=False,
                    )
                )
        if layer not in primary_by_layer:
            # Candidate layer has no other primaries
            hooks.append(
                make_full_z_injection_hook(
                    layer,
                    head,
                    z_head,
                    zero_heads=(),
                    zero_candidate=zero_candidate,
                )
            )
    else:
        # Clean context: only touch candidate head
        hooks.append(
            make_full_z_injection_hook(
                layer,
                head,
                z_head,
                zero_heads=(),
                zero_candidate=zero_candidate,
            )
        )

    # One hook per hook-point name
    return merge_hook_lists(hooks)
