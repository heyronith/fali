"""EF-1 Level B tiny HookedTransformer validation tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from extended_fali_synthetic_transformer import (  # noqa: E402
    EF1_SEED,
    build_level_b_context,
    endpoint_errors,
    no_live_leak_error,
    quotient_bridge_error,
    reconstruction_errors,
)

FP32_TOL = 2e-6
ENDPOINT_TOL = 1e-5
LEAK_TOL = 1e-6
QUOTIENT_TOL = 1e-5


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


def test_endpoint_identities(level_b_ctx):
    err = endpoint_errors(level_b_ctx)
    assert err["u00000_vs_intact"] <= ENDPOINT_TOL
    assert err["u11111_vs_conditional"] <= ENDPOINT_TOL
    assert err["gain_vs_delta_u"] <= ENDPOINT_TOL


def test_no_live_leak(level_b_ctx):
    assert no_live_leak_error(level_b_ctx) <= LEAK_TOL


def test_quotient_bridge(level_b_ctx):
    assert quotient_bridge_error(level_b_ctx) <= QUOTIENT_TOL


def test_architecture_seed(level_b_ctx):
    assert level_b_ctx.model.cfg.n_layers == 2
    assert level_b_ctx.model.cfg.n_heads == 2
    assert level_b_ctx.model.cfg.d_model == 32
