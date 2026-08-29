"""Tests for EF-5R zero-reference diagnosis."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from extended_fali_ef4_analysis import observed_canonical_balanced_mean  # noqa: E402
from extended_fali_ef5r_diagnosis import (  # noqa: E402
    TOL,
    endpoint_metrics_from_cells,
    extract_cell_state,
    classify_diagnosis,
    informative_label_clarification_md,
    io_semantic_audit,
    summarize_damage_by_pattern,
)
from historical_integrity_audit import (  # noqa: E402
    audit_ef5_historical_integrity,
    compare_git_blob_across_refs,
    tracked_git_blob_pass,
)


def _synthetic_cells(n_ex=2, seed=42):
    rows = []
    for ex in range(n_ex):
        for head in ("L9H7", "L9H0", "L10H2", "L10H10"):
            for mask in range(32):
                t = (mask >> 0) & 1
                r = (mask >> 1) & 1
                o = (mask >> 2) & 1
                v = (mask >> 3) & 1
                c = (mask >> 4) & 1
                clean = 1.0 + 0.01 * ex
                damaged = 0.5 + 0.01 * ex
                inj = clean if (t, r, o, v, c) == (0, 0, 0, 0, 0) else damaged
                rows.append(
                    {
                        "seed": seed,
                        "example_idx": ex,
                        "head_name": head,
                        "layer": 10,
                        "head": 2,
                        "T": t,
                        "R": r,
                        "O": o,
                        "V": v,
                        "C": c,
                        "metric_injected": inj,
                        "metric_zero": 0.0,
                        "U": inj,
                    }
                )
    return pd.DataFrame(rows)


def test_extract_all_a_all_b_cells():
    cells = _synthetic_cells()
    row = extract_cell_state(cells, example_idx=0, head_name="L10H2", state=(0, 0, 0, 0, 0))
    assert row["metric_injected"] == pytest.approx(1.0)
    row_b = extract_cell_state(cells, example_idx=0, head_name="L10H2", state=(1, 1, 1, 1, 1))
    assert row_b["metric_injected"] == pytest.approx(0.5)


def test_per_head_endpoint_agreement():
    ep = endpoint_metrics_from_cells(_synthetic_cells())
    for ex in (0, 1):
        sub = ep[ep["example_idx"] == ex]
        assert sub["clean_IO_minus_S"].nunique() == 1
        assert sub["damaged_IO_minus_S"].nunique() == 1


def test_zero_drop_reconstruction():
    ep = endpoint_metrics_from_cells(_synthetic_cells())
    assert (ep["drop_from_cells"] == ep["clean_IO_minus_S"] - ep["damaged_IO_minus_S"]).all()


def test_cross_scheme_clean_identity_synthetic():
    ep = endpoint_metrics_from_cells(_synthetic_cells())
    for ex in (0, 1):
        sub = ep[ep["example_idx"] == ex]
        assert sub["clean_IO_minus_S"].nunique() == 1


def test_matched_damage_drop_from_cells():
    ep = endpoint_metrics_from_cells(_synthetic_cells())
    assert (ep["drop_from_cells"] == ep["clean_IO_minus_S"] - ep["damaged_IO_minus_S"]).all()


def test_baba_abba_aggregation_structure():
    rows = []
    for seed in (42, 1, 8, 22):
        for tid in range(15):
            for pat in ("BABA", "ABBA"):
                for _ in range(30):
                    rows.append(
                        {
                            "damage_scheme": "zero",
                            "damage_drop": 1.0 if pat == "BABA" else -1.0,
                            "seed": seed,
                            "low_level_template_id": tid,
                            "pattern_type": pat,
                        }
                    )
    df = pd.DataFrame(rows)
    out = summarize_damage_by_pattern(df, "zero")
    assert set(out["pattern_type"]) == {"BABA", "ABBA"}


def test_fractional_criterion_not_applicable(monkeypatch):
    monkeypatch.setattr(
        "extended_fali_ef5r_diagnosis.diagnostic_ci",
        lambda *args, **kwargs: (0.1, 0.2),
    )
    rows = []
    for seed in (42, 1, 8, 22):
        for tid in range(15):
            for pat in ("BABA", "ABBA"):
                for _ in range(30):
                    rows.append(
                        {
                            "damage_scheme": "zero",
                            "damage_drop": -0.1,
                            "seed": seed,
                            "low_level_template_id": tid,
                            "pattern_type": pat,
                        }
                    )
                    rows.append(
                        {
                            "damage_scheme": "matched_mean",
                            "damage_drop": 0.5,
                            "seed": seed,
                            "low_level_template_id": tid,
                            "pattern_type": pat,
                        }
                    )
    damage_df = pd.DataFrame(rows)
    pat_df = summarize_damage_by_pattern(damage_df, "zero")
    md = informative_label_clarification_md(damage_df, pat_df)
    assert "NOT_APPLICABLE" in md


def test_diagnosis_category_heterogeneity():
    recon = {"implementation_inconsistency_stop": False}
    pat = pd.DataFrame(
        {
            "scheme": ["zero", "zero"],
            "pattern_type": ["BABA", "ABBA"],
            "exact_balanced_mean": [0.5, -0.5],
        }
    )
    assert classify_diagnosis(recon, pat) == "GENUINE_ZERO_HETEROGENEITY"


def test_historical_audit_not_tautological():
    repo = REPO
    cmp = compare_git_blob_across_refs(
        repo,
        ["results/extended_fali/ef4_canonical/aggregate/fine_score_gate.json"],
        "79feba9d9a97c62ada673159aaa1d13f757da05b",
        "5924f073fbdc703db8ae6f9d6b658eea1966c6fc",
    )
    assert "comparisons" in cmp
    assert cmp["comparisons"][0]["ref_a"] != cmp["comparisons"][0]["ref_b"]
    assert "all_match" not in cmp


def test_balanced_zero_mean_equals_half_pattern_means():
    rows = []
    for seed in (42, 1, 8, 22):
        for tid in range(15):
            for pat, val in (("BABA", 1.0), ("ABBA", -1.0)):
                for _ in range(30):
                    rows.append(
                        {
                            "damage_scheme": "zero",
                            "damage_drop": val + 0.01 * tid,
                            "seed": seed,
                            "low_level_template_id": tid,
                            "pattern_type": pat,
                        }
                    )
    damage_df = pd.DataFrame(rows)
    pat_df = summarize_damage_by_pattern(damage_df, "zero")
    baba = float(pat_df[pat_df["pattern_type"] == "BABA"]["exact_balanced_mean"].iloc[0])
    abba = float(pat_df[pat_df["pattern_type"] == "ABBA"]["exact_balanced_mean"].iloc[0])
    zero_rows = damage_df[damage_df["damage_scheme"] == "zero"]
    balanced = observed_canonical_balanced_mean(
        zero_rows["damage_drop"].values,
        zero_rows["seed"].values,
        zero_rows["low_level_template_id"].values,
        zero_rows["pattern_type"].values,
    )
    assert balanced == pytest.approx((baba + abba) / 2)
    assert balanced != pytest.approx(baba + abba)


@pytest.mark.skipif(not (REPO / "results/RAW_DATA_INDEX.json").exists(), reason="needs RAW_DATA_INDEX")
def test_historical_audit_raw_index_and_tracked_git():
    audit = audit_ef5_historical_integrity(REPO)
    assert "tracked_git_blob_pass" in audit
    assert "untracked_raw_index_audit_pass" in audit
    assert "per_seed_prepost_pass" in audit
    assert audit["untracked_raw_index_audit_pass"] is True
    assert audit["tracked_git_blob_pass"] == tracked_git_blob_pass(audit["git_blob_compare"])
    assert audit["overall_pass"] == (
        audit["tracked_git_blob_pass"]
        and audit["untracked_raw_index_audit_pass"]
        and audit["per_seed_prepost_pass"]
    )
    assert audit["raw_data_index_factorial"]["entries"]


def test_io_semantic_audit_distinguishes_position_and_token_identity():
    audit = io_semantic_audit(REPO)
    assert audit["position_order_audit"]["status"] == "PASS"
    assert audit["token_identity_audit"]["direct_per_example_token_id_revalidation"] is False
    assert audit["token_identity_audit"]["status"] == "INHERITED"


@pytest.mark.skipif(not (REPO / "results/extended_fali/ef5_ablation_robustness").exists(), reason="needs EF-5 results")
def test_ef5r_integration_smoke():
    from extended_fali_ef5r_diagnosis import run_diagnosis

    out = REPO / "results/extended_fali/ef5_reference_diagnosis_test"
    res = run_diagnosis(REPO, out)
    assert res["diagnosis"]["reconstruction_checks"]["check_C_ef5_forward_agreement"]
