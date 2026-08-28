"""
Tests for Revision Phase 2A/3A FALI + score/pattern/Q/K interventions.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from revision_fali import (  # noqa: E402
    attention_margin,
    build_hybrid_score_row,
    causal_key_mask,
    competing_logsumexp,
    extract_target_and_competing,
    fali_shapley,
    sigmoid_margin_prob,
    softmax_target_from_scores,
)
from revision_score_interventions import (  # noqa: E402
    compose_hooks,
    make_k_patch_hook,
    make_pattern_patch_hook,
    make_q_patch_hook,
    make_score_hybrid_hook,
    merge_hook_lists,
)
from revision_interventions import build_zero_ablation_hooks  # noqa: E402


# ---------------------------------------------------------------------------
# A. FALI algebra
# ---------------------------------------------------------------------------


def test_fali_algebra_identity():
    t_A, t_B = 1.0, 2.5
    b_A = torch.tensor([0.0, -1.0, 0.5], dtype=torch.float64)
    b_B = torch.tensor([-0.5, 1.0, 0.2], dtype=torch.float64)
    r = fali_shapley(t_A, b_A, t_B, b_B)
    assert abs(r.phi_target + r.phi_competing - r.total_delta_attn) < 1e-12
    assert abs(r.residual) < 1e-12
    assert abs(r.total_delta_attn - (r.p11 - r.p00)) < 1e-12


# ---------------------------------------------------------------------------
# B. Softmax-margin identity
# ---------------------------------------------------------------------------


def test_softmax_margin_identity():
    t = torch.tensor(1.7, dtype=torch.float64)
    b = torch.tensor([0.2, -0.5, 1.1, -2.0], dtype=torch.float64)
    p = softmax_target_from_scores(t, b)
    m = attention_margin(t, b)
    assert torch.allclose(p, sigmoid_margin_prob(m), atol=1e-12, rtol=0)


# ---------------------------------------------------------------------------
# C / D. Mask preservation + hybrid semantics
# ---------------------------------------------------------------------------


def test_hybrid_modes_and_mask_preservation():
    seq = 8
    q = 5
    io = 2
    a = torch.randn(seq)
    b = torch.randn(seq)
    # Causal mask: future = -inf
    a[q + 1 :] = float("-inf")
    b[q + 1 :] = float("-inf")

    f00 = build_hybrid_score_row(a, b, io, q, "F00")
    f10 = build_hybrid_score_row(a, b, io, q, "F10")
    f01 = build_hybrid_score_row(a, b, io, q, "F01")
    f11 = build_hybrid_score_row(a, b, io, q, "F11")

    valid = causal_key_mask(seq, q)
    assert torch.equal(~torch.isfinite(f00), ~torch.isfinite(b))
    assert torch.equal(~torch.isfinite(f10), ~torch.isfinite(b))
    assert torch.equal(~torch.isfinite(f01), ~torch.isfinite(b))
    assert torch.equal(~torch.isfinite(f11), ~torch.isfinite(b))

    assert torch.allclose(f00[valid], a[valid])
    assert torch.allclose(f11[valid], b[valid])
    assert torch.allclose(f10[io], b[io])
    bg = valid.clone()
    bg[io] = False
    assert torch.allclose(f10[bg], a[bg])
    assert torch.allclose(f01[io], a[io])
    assert torch.allclose(f01[bg], b[bg])

    # Future positions untouched (= B's -inf)
    assert torch.equal(f00[~valid], b[~valid])


def test_extract_rejects_masked_target():
    row = torch.tensor([1.0, 2.0, float("-inf"), 0.5])
    with pytest.raises(AssertionError):
        extract_target_and_competing(row, target_key=2, query_pos=3)


# ---------------------------------------------------------------------------
# E. Output Shapley algebra (numeric)
# ---------------------------------------------------------------------------


def test_output_shapley_algebra():
    F00, F10, F01, F11 = 1.0, 1.5, 0.8, 1.2
    causal_target = 0.5 * ((F10 - F00) + (F11 - F01))
    causal_competing = 0.5 * ((F01 - F00) + (F11 - F10))
    assert abs(causal_target + causal_competing - (F11 - F00)) < 1e-12


# ---------------------------------------------------------------------------
# F–I. Hook safety
# ---------------------------------------------------------------------------


def test_score_hook_clones_and_scopes():
    B, H, S = 2, 12, 6
    scores = torch.randn(B, H, S, S)
    # causal-ish: upper triangle -inf
    for q in range(S):
        scores[:, :, q, q + 1 :] = float("-inf")
    scores_A = scores + 0.5
    end = torch.tensor([3, 4])
    io = torch.tensor([1, 2])
    head = 2
    original = scores.clone()

    _, fn = make_score_hybrid_hook(scores_A, layer=10, head=head, end_positions=end, io_positions=io, mode="F10")
    out = fn(scores, hook=None)
    assert torch.equal(scores, original), "must not mutate input"
    # Unrelated heads unchanged
    for h in range(H):
        if h == head:
            continue
        assert torch.equal(out[:, h], original[:, h])
    # Unrelated query positions for candidate head unchanged
    for i in range(B):
        q = int(end[i])
        for qq in range(S):
            if qq == q:
                continue
            assert torch.equal(out[i, head, qq], original[i, head, qq])


def test_pattern_patch_scopes():
    B, H, S = 2, 12, 5
    pat = torch.rand(B, H, S, S)
    pat = pat / pat.sum(dim=-1, keepdim=True)
    pat_A = torch.rand(B, H, S, S)
    pat_A = pat_A / pat_A.sum(dim=-1, keepdim=True)
    end = torch.tensor([2, 3])
    head = 7
    original = pat.clone()
    _, fn = make_pattern_patch_hook(pat_A, 9, head, end)
    out = fn(pat, hook=None)
    assert torch.equal(pat, original)
    for h in range(H):
        if h != head:
            assert torch.equal(out[:, h], original[:, h])
    batch = torch.arange(B)
    assert torch.allclose(out[batch, head, end], pat_A[batch, head, end])


def test_q_patch_scopes():
    B, S, H, D = 2, 6, 12, 8
    q = torch.randn(B, S, H, D)
    q_A = torch.randn(B, S, H, D)
    end = torch.tensor([2, 4])
    head = 1
    original = q.clone()
    _, fn = make_q_patch_hook(q_A, 10, head, end)
    out = fn(q, hook=None)
    assert torch.equal(q, original)
    batch = torch.arange(B)
    assert torch.allclose(out[batch, end, head], q_A[batch, end, head])
    # other heads at end unchanged
    for h in range(H):
        if h == head:
            continue
        assert torch.equal(out[batch, end, h], original[batch, end, h])


def test_k_patch_only_valid_keys():
    B, S, H, D = 2, 6, 12, 8
    k = torch.randn(B, S, H, D)
    k_A = torch.randn(B, S, H, D)
    end = torch.tensor([2, 3])
    head = 5
    original = k.clone()
    _, fn = make_k_patch_hook(k_A, 10, head, end)
    out = fn(k, hook=None)
    assert torch.equal(k, original)
    for i in range(B):
        q = int(end[i])
        valid = causal_key_mask(S, q)
        assert torch.allclose(out[i, valid, head], k_A[i, valid, head])
        assert torch.equal(out[i, ~valid, head], original[i, ~valid, head])
        for h in range(H):
            if h == head:
                continue
            assert torch.equal(out[i, :, h], original[i, :, h])


def test_compose_one_hook_per_point():
    z_hooks = build_zero_ablation_hooks([(9, 6), (9, 9)])
    # Fake second z hook on same layer
    def other(z, hook):
        return z.clone()

    merged = merge_hook_lists(z_hooks, [("blocks.9.attn.hook_z", other)])
    names = [n for n, _ in merged]
    assert names.count("blocks.9.attn.hook_z") == 1
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# J. Dataset alignment helper (smoke metadata if present)
# ---------------------------------------------------------------------------


def test_phase1_smoke_metadata_exists_or_skip():
    path = REPO_ROOT / "results/revision_phase1/smoke/dataset_metadata.csv"
    if not path.exists():
        pytest.skip("Phase-1 smoke metadata not present")
    df = pd.read_csv(path)
    for c in ["example_idx", "prompt", "S", "IO", "template_id", "end_position"]:
        assert c in df.columns
    assert len(df) == 8


# ---------------------------------------------------------------------------
# L. No historical overwrite targets still present / hashed
# ---------------------------------------------------------------------------


def test_historical_phase1_raw_not_deleted():
    # Presence check only — experiment must not delete these
    p = REPO_ROOT / "results/revision_phase1/full/seed_42/candidate_raw.csv"
    if p.exists():
        assert p.stat().st_size > 0


def test_phase1c_stable_hash_not_python_hash():
    """Regression: bootstrap seeding must use SHA256, not hash()."""
    path = REPO_ROOT / "experiments/revision_phase1_corrected_classification.py"
    text = path.read_text()
    assert "stable_str_hash_int" in text
    assert "hashlib.sha256" in text
    # Ensure we don't seed with Python's hash()
    assert "hash(hname)" not in text
    assert "hash(head_name)" not in text


def test_phase1c_l10h1_suggestive_sign_flip():
    path = (
        REPO_ROOT
        / "results/revision_phase1/full/aggregate_corrected/functional_classification.csv"
    )
    if not path.exists():
        pytest.skip("corrected aggregate missing")
    df = pd.read_csv(path)
    row = df[df["head_name"] == "L10H1"].iloc[0]
    assert row["sign_flip_type"] == "suggestive_suppressive_to_supportive"
