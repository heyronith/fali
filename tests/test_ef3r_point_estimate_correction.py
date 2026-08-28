"""EF-3R tests: observed balanced mean vs bootstrap point estimate."""

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
    eval_h1,
    eval_h2_h3,
    go_stop_decision,
    load_merged_attributions,
    observed_balanced_mean,
    ro_discovery_eval,
    run_analysis,
    seed_balanced_means,
)


def _synthetic_cohort_values(n_per_seed: int = 200):
    seeds = np.array([42] * n_per_seed + [1] * n_per_seed + [8] * n_per_seed + [22] * n_per_seed)
    tmpl_block = np.array(["A"] * 100 + ["B"] * 100)
    templates = np.concatenate([tmpl_block, tmpl_block, tmpl_block, tmpl_block])
    return seeds, templates


def test_observed_balanced_mean_equals_eight_cell_means():
    rng = np.random.default_rng(7)
    values = rng.normal(size=800)
    seeds, templates = _synthetic_cohort_values()
    cell_means = []
    for s in EF3_SEEDS:
        for t in ("A", "B"):
            idx = np.where((seeds == s) & (templates == t))[0]
            cell_means.append(float(values[idx].mean()))
    direct = float(np.mean(cell_means))
    assert observed_balanced_mean(values, seeds, templates) == pytest.approx(direct)


def test_observed_point_independent_of_n_bootstrap():
    rng = np.random.default_rng(11)
    values = rng.normal(size=800)
    seeds, templates = _synthetic_cohort_values()
    point = observed_balanced_mean(values, seeds, templates)
    for n_boot in (100, 10000):
        boot = balanced_bootstrap_distribution(values, seeds, templates, n_bootstrap=n_boot)
        bootstrap_ci(boot)
        assert observed_balanced_mean(values, seeds, templates) == pytest.approx(point)


def test_bootstrap_distribution_deterministic():
    rng = np.random.default_rng(0)
    values = rng.normal(size=800)
    seeds, templates = _synthetic_cohort_values()
    b1 = balanced_bootstrap_distribution(values, seeds, templates, n_bootstrap=100)
    b2 = balanced_bootstrap_distribution(values, seeds, templates, n_bootstrap=100)
    assert np.allclose(b1, b2)


def test_bootstrap_tag_changes_ci_not_observed_point():
    rng = np.random.default_rng(3)
    values = rng.normal(size=800)
    seeds, templates = _synthetic_cohort_values()
    point = observed_balanced_mean(values, seeds, templates)
    boot_a = balanced_bootstrap_distribution(values, seeds, templates, n_bootstrap=500, seed_tag="tag-a")
    boot_b = balanced_bootstrap_distribution(values, seeds, templates, n_bootstrap=500, seed_tag="tag-b")
    lo_a, hi_a = bootstrap_ci(boot_a)
    lo_b, hi_b = bootstrap_ci(boot_b)
    assert observed_balanced_mean(values, seeds, templates) == pytest.approx(point)
    assert lo_a != pytest.approx(lo_b) or hi_a != pytest.approx(hi_b)


def test_ro_threshold_uses_observed_phi_score_mean():
    values = np.full(800, 0.5)
    seeds, templates = _synthetic_cohort_values()
    phi_score_obs = observed_balanced_mean(values, seeds, templates)
    res = ro_discovery_eval(values, seeds, templates, phi_score_mean=phi_score_obs, channel="phi_R")
    assert res["threshold"] == pytest.approx(0.10 * abs(phi_score_obs))


@pytest.fixture(scope="module")
def frozen_ef3_df():
    agg = REPO / "results/extended_fali/ef3_two_template/aggregate"
    if not (REPO / "results/extended_fali/ef3_two_template/seed_42/attribution_raw.csv").exists():
        pytest.skip("EF-3 attribution raw not available")
    return load_merged_attributions(agg, REPO)


def test_frozen_dataset_hypothesis_status_unchanged(frozen_ef3_df):
    df = frozen_ef3_df
    h1 = eval_h1(df)
    h2 = eval_h2_h3(
        df,
        "L10H2",
        ("conditional_gain", "Phi_SCORE", "phi_T"),
        {"conditional_gain": "positive", "Phi_SCORE": "positive", "phi_T": "positive"},
    )
    h3 = eval_h2_h3(
        df,
        "L10H10",
        ("Phi_SCORE", "phi_C", "conditional_gain", "phi_T"),
        {"Phi_SCORE": "positive", "phi_C": "negative", "conditional_gain": "negative", "phi_T": "positive"},
    )
    ro = {}
    for head in ("L10H2", "L10H10"):
        sub = df[df["head_name"] == head]
        phi_score_mean = observed_balanced_mean(
            sub["Phi_SCORE"].values, sub["seed"].values, sub["template_id"].values
        )
        for ch in ("phi_R", "phi_O"):
            ro[f"{head}_{ch}"] = ro_discovery_eval(
                sub[ch].values, sub["seed"].values, sub["template_id"].values, phi_score_mean, ch
            )
    go = go_stop_decision(True, h1, ro)
    assert h1["L9H7"]["architectural_pass"]
    assert all(h2["components"][c]["supported"] for c in ("conditional_gain", "Phi_SCORE", "phi_T"))
    assert all(h3["components"][c]["supported"] for c in ("Phi_SCORE", "phi_C", "conditional_gain", "phi_T"))
    assert ro["L10H2_phi_R"]["discovery_candidate"]
    assert ro["L10H10_phi_R"]["discovery_candidate"]
    assert not ro["L10H2_phi_O"]["discovery_candidate"]
    assert not ro["L10H10_phi_O"]["discovery_candidate"]
    assert go["decision"] == "GO"


def test_ef3r_audit_cross_checks(frozen_ef3_df):
    df = frozen_ef3_df
    checks = {
        ("L9H7", "phi_C"): 0.0720376,
        ("L9H0", "phi_C"): 0.00713691,
        ("L10H2", "conditional_gain"): 0.200339,
        ("L10H2", "Phi_SCORE"): 0.323728,
        ("L10H2", "phi_T"): 0.666720,
        ("L10H2", "phi_R"): -0.343628,
        ("L10H10", "Phi_SCORE"): 0.222209,
        ("L10H10", "phi_C"): -0.314975,
        ("L10H10", "conditional_gain"): -0.0861536,
        ("L10H10", "phi_T"): 0.374054,
        ("L10H10", "phi_R"): -0.149499,
    }
    for (head, metric), expected in checks.items():
        sub = df[df["head_name"] == head]
        got = observed_balanced_mean(sub[metric].values, sub["seed"].values, sub["template_id"].values)
        assert got == pytest.approx(expected, rel=0, abs=5e-4), f"{head} {metric}: {got} vs {expected}"


def test_run_analysis_produces_correction_report():
    agg = REPO / "results/extended_fali/ef3_two_template/aggregate"
    if not (agg / "bootstrap_ci.csv").exists():
        pytest.skip("EF-3 aggregate not available")
    out = run_analysis(REPO, agg)
    assert (agg / "analysis_correction_report.json").exists()
    assert (agg / "ef4_candidate_hypotheses.json").exists()
    assert out["correction_status_stable"]
    corr = __import__("json").loads((agg / "analysis_correction_report.json").read_text())
    assert corr["status_stable"]
    assert corr["max_absolute_point_shift"] > 0
