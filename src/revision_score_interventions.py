"""
Clone-safe attention-score / pattern / Q / K interventions for Phase-2/3 revision.

All hooks clone before mutation. Multiple ops on the same hook point are composed
into a single hook (never rely on fragile hook registration order).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from revision_fali import build_hybrid_score_row, causal_key_mask
from revision_interventions import HookFn, HookSpec, build_zero_ablation_hooks

Head = Tuple[int, int]


def compose_hooks(hooks: Sequence[HookSpec]) -> List[HookSpec]:
    """
    Merge hooks that share a hook-point name into one composed clone-safe hook.
    Composition order: registration / list order (left-to-right).
    """
    grouped: Dict[str, List[HookFn]] = defaultdict(list)
    order: List[str] = []
    for name, fn in hooks:
        if name not in grouped:
            order.append(name)
        grouped[name].append(fn)

    out: List[HookSpec] = []
    for name in order:
        fns = grouped[name]
        if len(fns) == 1:
            out.append((name, fns[0]))
            continue

        def _make_composed(functions: List[HookFn]) -> HookFn:
            def composed(x: torch.Tensor, hook) -> torch.Tensor:
                out_t = x
                for f in functions:
                    out_t = f(out_t, hook)
                return out_t

            return composed

        out.append((name, _make_composed(list(fns))))
    return out


def merge_hook_lists(*hook_lists: Sequence[HookSpec]) -> List[HookSpec]:
    flat: List[HookSpec] = []
    for hl in hook_lists:
        flat.extend(list(hl))
    return compose_hooks(flat)


def _batch_index(batch: int, device) -> torch.Tensor:
    return torch.arange(batch, device=device)


def make_score_hybrid_hook(
    scores_A: torch.Tensor,
    layer: int,
    head: int,
    end_positions: torch.Tensor,
    io_positions: torch.Tensor,
    mode: str,
) -> HookSpec:
    """
    Replace candidate head's pre-softmax score row at query=end with a hybrid
    of clean A and live B scores. Modes: F00, F10, F01, F11.
    """
    hook_name = f"blocks.{layer}.attn.hook_attn_scores"
    scores_A = scores_A.detach()

    def hook_fn(scores: torch.Tensor, hook) -> torch.Tensor:
        # scores: [batch, n_heads, q, k]
        assert scores.ndim == 4
        device = scores.device
        out = scores.clone()
        B = out.shape[0]
        batch_idx = _batch_index(B, device)
        end = end_positions.to(device)
        io = io_positions.to(device)
        a = scores_A.to(device)

        for i in range(B):
            q = int(end[i].item())
            tkey = int(io[i].item())
            hybrid = build_hybrid_score_row(
                a[i, head, q, :],
                out[i, head, q, :],
                target_key=tkey,
                query_pos=q,
                mode=mode,
            )
            # Only touch this head + query row
            out[i, head, q, :] = hybrid
        return out

    return hook_name, hook_fn


def make_pattern_patch_hook(
    pattern_A: torch.Tensor,
    layer: int,
    head: int,
    end_positions: torch.Tensor,
) -> HookSpec:
    """Patch candidate attention PATTERN row at query=end from clean A."""
    hook_name = f"blocks.{layer}.attn.hook_pattern"
    pattern_A = pattern_A.detach()

    def hook_fn(pattern: torch.Tensor, hook) -> torch.Tensor:
        # pattern: [batch, n_heads, q, k]
        out = pattern.clone()
        device = out.device
        B = out.shape[0]
        batch_idx = _batch_index(B, device)
        end = end_positions.to(device)
        a = pattern_A.to(device)
        out[batch_idx, head, end, :] = a[batch_idx, head, end, :]
        return out

    return hook_name, hook_fn


def make_q_patch_hook(
    q_A: torch.Tensor,
    layer: int,
    head: int,
    end_positions: torch.Tensor,
) -> HookSpec:
    """Patch candidate q vector at query=end from clean A."""
    hook_name = f"blocks.{layer}.attn.hook_q"
    q_A = q_A.detach()

    def hook_fn(q: torch.Tensor, hook) -> torch.Tensor:
        # q: [batch, seq, n_heads, d_head]
        out = q.clone()
        device = out.device
        B = out.shape[0]
        batch_idx = _batch_index(B, device)
        end = end_positions.to(device)
        a = q_A.to(device)
        out[batch_idx, end, head, :] = a[batch_idx, end, head, :]
        return out

    return hook_name, hook_fn


def make_k_patch_hook(
    k_A: torch.Tensor,
    layer: int,
    head: int,
    end_positions: torch.Tensor,
) -> HookSpec:
    """
    Patch candidate k vectors for causally valid source positions from clean A.

    For each example i, replace K at positions k <= end[i] for the candidate head.
    Future keys are left unchanged (and should remain unused under causal attn).
    """
    hook_name = f"blocks.{layer}.attn.hook_k"
    k_A = k_A.detach()

    def hook_fn(k: torch.Tensor, hook) -> torch.Tensor:
        # k: [batch, seq, n_heads, d_head]
        out = k.clone()
        device = out.device
        B, seq, _, _ = out.shape
        end = end_positions.to(device)
        a = k_A.to(device)
        for i in range(B):
            q = int(end[i].item())
            valid = causal_key_mask(seq, q, device=device)
            out[i, valid, head, :] = a[i, valid, head, :]
        return out

    return hook_name, hook_fn


def primary_ablation_hooks() -> List[HookSpec]:
    from revision_config import PRIMARY_NAME_MOVERS

    return build_zero_ablation_hooks(PRIMARY_NAME_MOVERS)


def hooks_for_score_hybrid(
    scores_A: torch.Tensor,
    layer: int,
    head: int,
    end_positions: torch.Tensor,
    io_positions: torch.Tensor,
    mode: str,
    include_primary_ablation: bool = True,
) -> List[HookSpec]:
    score_hook = make_score_hybrid_hook(
        scores_A, layer, head, end_positions, io_positions, mode
    )
    if include_primary_ablation:
        return merge_hook_lists(primary_ablation_hooks(), [score_hook])
    return [score_hook]


def hooks_for_pattern_patch(
    pattern_A: torch.Tensor,
    layer: int,
    head: int,
    end_positions: torch.Tensor,
    include_primary_ablation: bool = True,
) -> List[HookSpec]:
    p_hook = make_pattern_patch_hook(pattern_A, layer, head, end_positions)
    if include_primary_ablation:
        return merge_hook_lists(primary_ablation_hooks(), [p_hook])
    return [p_hook]


def hooks_for_q_patch(
    q_A: torch.Tensor,
    layer: int,
    head: int,
    end_positions: torch.Tensor,
    include_primary_ablation: bool = True,
) -> List[HookSpec]:
    q_hook = make_q_patch_hook(q_A, layer, head, end_positions)
    if include_primary_ablation:
        return merge_hook_lists(primary_ablation_hooks(), [q_hook])
    return [q_hook]


def hooks_for_k_patch(
    k_A: torch.Tensor,
    layer: int,
    head: int,
    end_positions: torch.Tensor,
    include_primary_ablation: bool = True,
) -> List[HookSpec]:
    k_hook = make_k_patch_hook(k_A, layer, head, end_positions)
    if include_primary_ablation:
        return merge_hook_lists(primary_ablation_hooks(), [k_hook])
    return [k_hook]


def hooks_for_qk_patch(
    q_A: torch.Tensor,
    k_A: torch.Tensor,
    layer: int,
    head: int,
    end_positions: torch.Tensor,
    include_primary_ablation: bool = True,
) -> List[HookSpec]:
    q_hook = make_q_patch_hook(q_A, layer, head, end_positions)
    k_hook = make_k_patch_hook(k_A, layer, head, end_positions)
    if include_primary_ablation:
        return merge_hook_lists(primary_ablation_hooks(), [q_hook, k_hook])
    return merge_hook_lists([q_hook, k_hook])
