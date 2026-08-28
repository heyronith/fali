"""Tests for EF-3 analysis, bootstrap, and reporting fixes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from extended_fali_ef3_analysis import (  # noqa: E402
    EF3_SEEDS,
    balanced_bootstrap_distribution,
    bootstrap_ci,
    direction_support,
    go_stop_decision,
    observed_balanced_mean,
    ro_discovery_eval,
    seed_balanced_means,
)
from extended_fali_gpt2 import owen_diagnostics, verify_historical_hash_integrity  # noqa: E402
from extended_fali_interventions import enumerate_extended_fali_cells  # noqa: E402


def test_ef3_seed_guard_constants():
    assert len(EF3_SEEDS) == 4
    assert 12 * 200 * 32 == 76800
    assert 12 * 200 * 32 * 4 == 307200


def _synthetic_cohort_values(n_per_seed: int = 200):
    seeds = np.array([42] * n_per_seed + [1] * n_per_seed + [8] * n_per_seed + [22] * n_per_seed)
    tmpl_block = np.array(["A"] * 100 + ["B"] * 100)
    templates = np.concatenate([tmpl_block, tmpl_block, tmpl_block, tmpl_block])
    return seeds, templates


def test_balanced_bootstrap_deterministic():
    n = 800
    rng = np.random.default_rng(0)
    values = rng.normal(size=n)
    seeds, templates = _synthetic_cohort_values()
    b1 = balanced_bootstrap_distribution(values, seeds, templates, n_bootstrap=100)
    b2 = balanced_bootstrap_distribution(values, seeds, templates, n_bootstrap=100)
    assert np.allclose(b1, b2)


def test_seed_balanced_means_structure():
    values = np.ones(800)
    seeds, templates = _synthetic_cohort_values()
    sm = seed_balanced_means(values, seeds, templates)
    assert len(sm) == 4
    assert all(v == 1.0 for v in sm.values())


def test_direction_support_positive():
    values = np.linspace(0.1, 1.0, 800)
    seeds, templates = _synthetic_cohort_values()
    res = direction_support(values, seeds, templates, "positive")
    assert res["ci_support"]
    assert res["seed_support"]
    assert res["supported"]


def test_ro_discovery_rule():
    values = np.full(800, 0.5)
    seeds, templates = _synthetic_cohort_values()
    res = ro_discovery_eval(values, seeds, templates, phi_score_mean=1.0, channel="phi_R")
    assert res["discovery_candidate"]


def test_go_stop_rule():
    h1 = {"L9H7_context_only_pass": True}
    ro = {"L10H2_phi_R": {"discovery_candidate": True}}
    g = go_stop_decision(True, h1, ro)
    assert g["decision"] == "GO"
    g2 = go_stop_decision(True, h1, {})
    assert g2["decision"] == "STOP"


def test_owen_diagnostics_per_block_residuals():
    table = {s: float(i) for i, (_, s) in enumerate(enumerate_extended_fali_cells())}
    diag = owen_diagnostics(table)
    assert "quotient_score_residual" in diag
    assert "quotient_value_residual" in diag
    assert "quotient_context_residual" in diag
    assert diag["quotient_score_residual"] <= diag["quotient_block_max"] + 1e-15


def test_historical_hash_missing_fails():
    before = {"path/a": {"sha256": None, "size": None}}
    after = {"path/a": {"sha256": None, "size": None}}
    ok, checks = verify_historical_hash_integrity(before, after, ["path/a"])
    assert not ok
    assert checks["path/a"] == "NOT_AVAILABLE"


def test_historical_hash_unchanged_passes():
    h = "abc123"
    before = {"path/a": {"sha256": h, "size": 1}}
    after = {"path/a": {"sha256": h, "size": 1}}
    ok, checks = verify_historical_hash_integrity(before, after, ["path/a"])
    assert ok
    assert checks["path/a"] == "UNCHANGED"
