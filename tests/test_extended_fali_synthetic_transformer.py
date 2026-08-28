"""EF-1 Level B tiny HookedTransformer validation tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from extended_fali_interventions import validate_score_partition  # noqa: E402
from extended_fali_synthetic_transformer import (  # noqa: E402
    EF1_SEED,
    build_level_b_context,
    endpoint_activation_errors,
    endpoint_errors,
    independent_endpoint_values,
    injection_scope_errors,
    live_leak_errors,
    quotient_bridge_error,
    reconstruction_errors,
)

FP32_TOL = 2e-6
ENDPOINT_TOL = 1e-5
LEAK_TOL = 1e-6
QUOTIENT_TOL = 1e-5
SCOPE_TOL = 1e-7


@pytest.fixture(scope="module")
def level_b_ctx():
    return build_level_b_context(seed=EF1_SEED)


def test_score_pattern_reconstruction(level_b_ctx):
    err = reconstruction_errors(level_b_ctx)
    assert err["pattern_a"] <= FP32_TOL
    assert err["pattern_b"] <= FP32_TOL
    assert err["z_a"] <= FP32_TOL
    assert err["z_b"] <= FP32_TOL
    assert err["masked_prob_max"] <= 1e-8


def test_independent_endpoint_identities(level_b_ctx):
    err = endpoint_errors(level_b_ctx)
    assert err["u00000_vs_intact"] <= ENDPOINT_TOL
    assert err["u11111_vs_conditional"] <= ENDPOINT_TOL
    assert err["gain_vs_delta_u"] <= ENDPOINT_TOL


def test_endpoint_activation_z(level_b_ctx):
    err = endpoint_activation_errors(level_b_ctx)
    assert err["all_a_z_error"] <= FP32_TOL
    assert err["all_b_z_error"] <= FP32_TOL


def test_injection_scope(level_b_ctx):
    err = injection_scope_errors(level_b_ctx)
    assert err["candidate_z_error"] <= FP32_TOL
    assert err["noncandidate_z_error"] <= SCOPE_TOL


def test_live_leak_v_q_qv(level_b_ctx):
    err = live_leak_errors(level_b_ctx)
    assert err["v_leak_error"] <= LEAK_TOL
    assert err["q_leak_error"] <= LEAK_TOL
    assert err["qv_leak_error"] <= LEAK_TOL


def test_quotient_bridge(level_b_ctx):
    assert quotient_bridge_error(level_b_ctx) <= QUOTIENT_TOL


def test_architecture_seed(level_b_ctx):
    assert level_b_ctx.model.cfg.n_layers == 2
    assert level_b_ctx.model.cfg.n_heads == 2
    assert level_b_ctx.model.cfg.d_model == 32


def test_batched_partition_different_target_keys():
    """Explicit structurally-valid target keys — infrastructure, not science."""
    batch = 2
    seq_len = 6
    final_query = torch.tensor([5, 5])
    target_keys = torch.tensor([1, 3])
    valid_lengths = torch.tensor([6, 6])
    ctx = build_level_b_context(
        seed=EF1_SEED,
        batch=batch,
        target_key_override=target_keys,
    )
    assert ctx.target_key.tolist() == [1, 3]
    validate_score_partition(ctx.partition)
    assert len(set(ctx.target_key.tolist())) >= 2
    vals = independent_endpoint_values(ctx)
    assert vals["u00000_vs_intact"] <= ENDPOINT_TOL
