"""Tests for Phase-3C full-head P×V×C factorial decomposition."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from revision_factorial import (  # noqa: E402
    build_factorial_fwd_hooks,
    make_full_z_injection_hook,
    max_abs_error,
    reconstruct_z_head,
    three_factor_shapley,
)


def test_three_factor_shapley_synthetic_additive():
    # Purely additive: H = a*P + b*V + c*C  (relative to H000=0)
    a, b, c = 1.5, -0.7, 2.0
    H000 = 0.0
    H100, H010, H001 = a, b, c
    H110, H101, H011 = a + b, a + c, b + c
    H111 = a + b + c
    phi_P, phi_V, phi_C, res = three_factor_shapley(
        H000, H100, H010, H110, H001, H101, H011, H111
    )
    assert abs(res) < 1e-12
    assert abs(phi_P - a) < 1e-12
    assert abs(phi_V - b) < 1e-12
    assert abs(phi_C - c) < 1e-12


def test_three_factor_shapley_with_interactions():
    # Non-additive cells
    cells = dict(
        H000=1.0,
        H100=2.0,
        H010=0.5,
        H110=3.5,
        H001=-1.0,
        H101=0.0,
        H011=1.0,
        H111=4.0,
    )
    phi_P, phi_V, phi_C, res = three_factor_shapley(
        cells["H000"],
        cells["H100"],
        cells["H010"],
        cells["H110"],
        cells["H001"],
        cells["H101"],
        cells["H011"],
        cells["H111"],
    )
    assert abs(res) < 1e-12
    assert abs(phi_P + phi_V + phi_C - (cells["H111"] - cells["H000"])) < 1e-12


def test_reconstruct_z_matches_einsum_definition():
    B, H, S, D = 2, 12, 5, 8
    pattern = torch.softmax(torch.randn(B, H, S, S), dim=-1)
    # causal-ish mask
    for q in range(S):
        pattern[:, :, q, q + 1 :] = 0
        pattern[:, :, q, : q + 1] = pattern[:, :, q, : q + 1] / pattern[:, :, q, : q + 1].sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
    v = torch.randn(B, S, H, D)
    head = 3
    z = reconstruct_z_head(pattern, v, head)
    # manual
    expected = torch.zeros(B, S, D)
    for b in range(B):
        for q in range(S):
            expected[b, q] = (pattern[b, head, q, :, None] * v[b, :, head, :]).sum(0)
    assert torch.allclose(z, expected, atol=1e-6)


def test_z_injection_clone_and_scope():
    B, S, H, D = 2, 6, 12, 8
    z = torch.randn(B, S, H, D)
    original = z.clone()
    replacement = torch.randn(B, S, D)
    _, fn = make_full_z_injection_hook(10, 2, replacement, zero_heads=(0,), zero_candidate=False)
    out = fn(z, hook=None)
    assert torch.equal(z, original)
    assert torch.allclose(out[:, :, 2, :], replacement)
    assert torch.count_nonzero(out[:, :, 0, :]) == 0
    for h in range(H):
        if h in (0, 2):
            continue
        assert torch.equal(out[:, :, h, :], original[:, :, h, :])


def test_zero_candidate_full_sequence():
    B, S, H, D = 1, 4, 12, 4
    z = torch.ones(B, S, H, D)
    _, fn = make_full_z_injection_hook(9, 7, None, zero_candidate=True)
    out = fn(z, hook=None)
    assert torch.count_nonzero(out[:, :, 7, :]) == 0
    assert torch.equal(out[:, :, 6, :], z[:, :, 6, :])


def test_build_hooks_one_per_layer_context_B():
    z = torch.randn(2, 5, 64)
    hooks = build_factorial_fwd_hooks(
        (10, 2), context_B=True, z_head=z, zero_candidate=False
    )
    names = [n for n, _ in hooks]
    assert len(names) == len(set(names))
    assert "blocks.9.attn.hook_z" in names
    assert "blocks.10.attn.hook_z" in names


def test_build_hooks_same_layer_primary_folded():
    z = torch.randn(2, 5, 64)
    hooks = build_factorial_fwd_hooks(
        (9, 7), context_B=True, z_head=z, zero_candidate=False
    )
    names = [n for n, _ in hooks]
    # Layer 9 should appear once (primaries + candidate composed)
    assert names.count("blocks.9.attn.hook_z") == 1
    assert "blocks.10.attn.hook_z" in names


def test_historical_phase1_untouched_exists():
    p = REPO / "results/revision_phase1/full/seed_42/candidate_raw.csv"
    if p.exists():
        assert p.stat().st_size > 0


def test_factorial_interaction_identity():
    from revision_factorial import factorial_interactions

    cells = dict(
        H000=1.0,
        H100=2.5,
        H010=0.5,
        H110=3.0,
        H001=-0.5,
        H101=1.2,
        H011=0.8,
        H111=4.0,
    )
    inter = factorial_interactions(
        cells["H000"],
        cells["H100"],
        cells["H010"],
        cells["H110"],
        cells["H001"],
        cells["H101"],
        cells["H011"],
        cells["H111"],
    )
    assert abs(inter["interaction_recon_residual"]) < 1e-12


def test_endpoint_marginal_definitions():
    from revision_factorial import endpoint_marginal_effects

    m = endpoint_marginal_effects(0, 1, 2, 3, 4, 5, 6, 7)
    assert abs(m["pattern_effect_B_VB"] - (7 - 6)) < 1e-12
    assert abs(m["value_effect_B_PB"] - (7 - 5)) < 1e-12
    assert abs(m["context_effect_PB_VB"] - (7 - 3)) < 1e-12
    assert abs(m["pattern_effect_A_VA"] - (1 - 0)) < 1e-12
