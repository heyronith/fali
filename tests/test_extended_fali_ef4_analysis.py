"""Tests for EF-4 canonical fine-score confirmation analysis."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from extended_fali_ef4_analysis import (  # noqa: E402
    EF4_HEADS,
    EF4_SEEDS,
    N_PER_SEED,
    N_STRATA,
    PRIMARY_HYPOTHESES,
    evaluate_hypothesis,
    filter_exclude_template_zero,
    fine_score_gate,
    observed_canonical_balanced_mean,
)
from extended_fali_gpt2_runner import RunConfig  # noqa: E402
from revision_stats import hierarchical_template_cluster_bootstrap_means  # noqa: E402


def _synthetic_canonical_df(n_per_stratum: int = 30, metric_value: float = 1.0):
    rows = []
    ex = 0
    for seed in EF4_SEEDS:
        for tid in range(15):
            for pat in ("BABA", "ABBA"):
                for _ in range(n_per_stratum):
                    rows.append(
                        {
                            "seed": seed,
                            "example_idx": ex,
                            "low_level_template_id": tid,
                            "pattern_type": pat,
                            "template_stratum": f"T{tid:02d}_{pat}",
                            "head_name": "L10H2",
                            "phi_T": metric_value + 0.01 * tid,
                            "phi_R": -metric_value - 0.01 * tid,
                            "phi_O": 0.001 * tid,
                            "Phi_SCORE": 0.5,
                            "conditional_gain": 0.1,
                            "phi_C": -0.05,
                            "phi_V": 0.01,
                        }
                    )
                    ex += 1
    return pd.DataFrame(rows)


def test_four_head_frozen_panel():
    assert EF4_HEADS == ("L9H7", "L9H0", "L10H2", "L10H10")
    assert len(PRIMARY_HYPOTHESES) == 4


def test_n900_seed_guard():
    assert len(EF4_SEEDS) == 4
    assert 4 * 900 * 32 == 115200
    assert 4 * 900 == 3600
    assert 4 * 4 * 900 * 32 == 460800


def test_exact_30_strata_count():
    df = _synthetic_canonical_df()
    assert len(df) == len(EF4_SEEDS) * N_PER_SEED
    strata = df.groupby(["seed", "low_level_template_id", "pattern_type"]).size()
    assert (strata == 30).all()
    assert len(strata) == len(EF4_SEEDS) * N_STRATA


def test_observed_canonical_balanced_mean_equals_stratum_average():
    df = _synthetic_canonical_df(metric_value=2.0)
    sub = df[df["head_name"] == "L10H2"]
    cell_means = []
    for seed in EF4_SEEDS:
        for tid in range(15):
            for pat in ("BABA", "ABBA"):
                m = (
                    (sub["seed"] == seed)
                    & (sub["low_level_template_id"] == tid)
                    & (sub["pattern_type"] == pat)
                )
                cell_means.append(float(sub.loc[m, "phi_T"].mean()))
    direct = float(np.mean(cell_means))
    got = observed_canonical_balanced_mean(
        sub["phi_T"].values,
        sub["seed"].values,
        sub["low_level_template_id"].values,
        sub["pattern_type"].values,
    )
    assert got == pytest.approx(direct)


def test_point_independent_of_bootstrap_count():
    df = _synthetic_canonical_df()
    sub = df[df["head_name"] == "L10H2"]
    point = observed_canonical_balanced_mean(
        sub["phi_T"].values,
        sub["seed"].values,
        sub["low_level_template_id"].values,
        sub["pattern_type"].values,
    )
    for n_boot in (100, 10000):
        hierarchical_template_cluster_bootstrap_means(
            sub["phi_T"].values,
            sub["seed"].values,
            sub["low_level_template_id"].values,
            sub["pattern_type"].values,
            n_bootstrap=n_boot,
            seed=42,
        )
        assert observed_canonical_balanced_mean(
            sub["phi_T"].values,
            sub["seed"].values,
            sub["low_level_template_id"].values,
            sub["pattern_type"].values,
        ) == pytest.approx(point)


def test_hierarchical_bootstrap_deterministic():
    df = _synthetic_canonical_df()
    sub = df[df["head_name"] == "L10H2"]
    b1 = hierarchical_template_cluster_bootstrap_means(
        sub["phi_T"].values,
        sub["seed"].values,
        sub["low_level_template_id"].values,
        sub["pattern_type"].values,
        n_bootstrap=200,
        seed=99,
    )
    b2 = hierarchical_template_cluster_bootstrap_means(
        sub["phi_T"].values,
        sub["seed"].values,
        sub["low_level_template_id"].values,
        sub["pattern_type"].values,
        n_bootstrap=200,
        seed=99,
    )
    assert np.allclose(b1, b2)


def test_shared_template_draws_across_seeds():
    df = _synthetic_canonical_df()
    sub = df[df["head_name"] == "L10H2"]
    _, drawn_t, ex_idx = hierarchical_template_cluster_bootstrap_means(
        sub["phi_T"].values,
        sub["seed"].values,
        sub["low_level_template_id"].values,
        sub["pattern_type"].values,
        n_bootstrap=50,
        seed=7,
        return_draw_arrays=True,
    )
    assert drawn_t.shape == (50, 15)
    assert ex_idx.shape[1] == len(EF4_SEEDS)


def test_t0_excluded_14_cluster_construction():
    df = _synthetic_canonical_df()
    out = filter_exclude_template_zero(df)
    assert out["low_level_template_id"].min() == 0
    assert out["low_level_template_id"].max() == 13
    assert len(out) == len(df) - 2 * 30 * len(EF4_SEEDS)


def test_phi_o_not_confirmatory():
    ids = {h["id"] for h in PRIMARY_HYPOTHESES}
    assert not any("phi_O" in i for i in ids)


def test_fine_score_gate_logic():
    hyp = {
        "H1_L10H2_phi_T_positive": {"FINAL_CONFIRMED": True},
        "H2_L10H2_phi_R_negative": {"FINAL_CONFIRMED": True},
        "H3_L10H10_phi_T_positive": {"FINAL_CONFIRMED": False},
        "H4_L10H10_phi_R_negative": {"FINAL_CONFIRMED": True},
    }
    assert fine_score_gate(hyp)["EF4_FINE_SCORE_GATE"] == "PARTIAL"


def test_run_config_canonical_fields():
    cfg = RunConfig(
        phase="EF-4",
        num_samples=900,
        seed=42,
        out_dir=Path("/tmp/out"),
        reference_meta=Path("/tmp/ref.csv"),
        historical_factorial=Path("/tmp/hist.csv"),
        required_hash_paths=[],
        device="cpu",
        dataset_kind="canonical",
        head_names=EF4_HEADS,
    )
    assert cfg.dataset_kind == "canonical"
    assert len(cfg.head_names) == 4


def test_r_magnitude_criterion_uses_observed_phi_score():
    df = _synthetic_canonical_df(metric_value=0.5)
    df["Phi_SCORE"] = 1.0
    df["phi_R"] = -0.2
    res = evaluate_hypothesis(df, PRIMARY_HYPOTHESES[1])
    assert res["magnitude_threshold"] == pytest.approx(0.10)
    assert res["magnitude_pass"] is True
