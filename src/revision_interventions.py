"""
Safe ablation hooks for the FALI revision pipeline.

Guarantees:
- no in-place mutation of activation tensors
- exactly one hook per layer hook_z point
- composite zeroing when multiple heads share a layer
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List, Sequence, Tuple

import torch

from revision_config import Head, assert_head_bounds

HookFn = Callable[[torch.Tensor, object], torch.Tensor]
HookSpec = Tuple[str, HookFn]


def _make_composite_zero_hook(head_indices: Sequence[int]) -> HookFn:
    head_indices = sorted(set(int(h) for h in head_indices))

    def hook_fn(z: torch.Tensor, hook) -> torch.Tensor:
        # z: [batch, seq_len, n_heads, d_head]
        assert z.ndim == 4, f"Expected hook_z rank-4 tensor, got shape {tuple(z.shape)}"
        n_heads = z.shape[2]
        for h in head_indices:
            assert 0 <= h < n_heads, f"Head index {h} invalid for n_heads={n_heads}"
        z_cloned = z.clone()
        for h in head_indices:
            z_cloned[:, :, h, :] = 0.0
        return z_cloned

    return hook_fn


def group_heads_by_layer(heads: Sequence[Head]) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for layer, head in heads:
        grouped[int(layer)].append(int(head))
    # Deduplicate while preserving sorted order for determinism
    return {layer: sorted(set(hs)) for layer, hs in sorted(grouped.items())}


def build_zero_ablation_hooks(
    heads: Sequence[Head],
    n_layers: int = 12,
    n_heads: int = 12,
) -> List[HookSpec]:
    """
    Build clone-safe zero-ablation hooks for an arbitrary set of (layer, head) pairs.

    Returns exactly one (hook_name, hook_fn) entry per distinct layer present in `heads`.
    """
    assert_head_bounds(heads, n_layers=n_layers, n_heads=n_heads)
    if not heads:
        return []

    grouped = group_heads_by_layer(heads)
    hooks: List[HookSpec] = []
    for layer, head_indices in grouped.items():
        hook_name = f"blocks.{layer}.attn.hook_z"
        hooks.append((hook_name, _make_composite_zero_hook(head_indices)))

    # Safety: one hook per hook point name
    names = [name for name, _ in hooks]
    assert len(names) == len(set(names)), f"Duplicate hook names generated: {names}"
    return hooks


def hook_point_names(hooks: Sequence[HookSpec]) -> List[str]:
    return [name for name, _ in hooks]
