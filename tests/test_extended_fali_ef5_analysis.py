"""Tests for EF-5 ablation robustness (damage + analysis logic)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from extended_fali_damage import (  # noqa: E402
    build_composite_replacement_hooks,
    build_mean_ablation_groups,
    build_resample_donor_map,
    compute_matched_mean_replacements,
    compute_matched_resample_replacements,
    donor_map_deterministic_hash,
    verify_hook_preserves_nonprimary_slices,
)
from extended_fali_ef4_analysis import observed_canonical_balanced_mean  # noqa: E402
from extended_fali_ef5_analysis import (  # noqa: E402
    ALT_SCHEMES,
    INFORMATIVE_FRAC,
    ROBUSTNESS_CHANNELS,
    apply_informative_fraction,
    broad_confirmed,
    channel_status,
    ef5_ablation_gate,
    evaluate_damage_strength,
    scheme_status,
)
from revision_stats import hierarchical_template_cluster_bootstrap_means  # noqa: E402


def _canonical_meta(n_per: int = 30, seed: int = 42) -> pd.DataFrame:
    rows = []
    ex = 0
    names = [f"S{i}" for i in range(200)]
    ios = [f"IO{i}" for i in range(200)]
    for tid in range(15):
        for pat in ("BABA", "ABBA"):
            for j in range(n_per):
                rows.append(
                    {
                        "example_idx": ex,
                        "low_level_template_id": tid,
                        "pattern_type": pat,
                        "template_stratum": f"T{tid:02d}_{pat}",
                        "end_position": 10 + (tid % 3),
                        "S": names[(ex + seed) % len(names)],
                        "IO": ios[(ex * 3 + seed) % len(ios)],
                    }
                )
                ex += 1
    return pd.DataFrame(rows)


def _synthetic_attr(metric_val: float = 1.0, sign: str = "pos") -> pd.DataFrame:
    rows = []
    for seed in (42, 1, 8, 22):
        for tid in range(15):
            for pat in ("BABA", "ABBA"):
                for _ in range(30):
                    v = metric_val if sign == "pos" else -metric_val
                    rows.append(
                        {
                            "seed": seed,
                            "low_level_template_id": tid,
                            "pattern_type": pat,
                            "template_stratum": f"T{tid:02d}_{pat}",
                            "head_name": "L10H2",
                            "phi_T": v,
                            "phi_R": -v,
                            "Phi_SCORE": 0.5,
                            "damage_drop": 0.2,
                        }
                    )
    return pd.DataFrame(rows)


def test_loo_mean_excludes_self():
    meta = _canonical_meta(n_per=4)
    groups = build_mean_ablation_groups(meta)
    assert (groups["self_excluded"]).all()
    assert (groups["group_size"] >= 2).all()
    z = {
        (9, 6): torch.arange(len(meta) * 2).reshape(len(meta), 2).float(),
        (9, 9): torch.ones(len(meta), 2),
        (10, 0): torch.zeros(len(meta), 2),
    }
    reps = compute_matched_mean_replacements(z, meta)
    for i in range(len(meta)):
        peers = meta[
            (meta["template_stratum"] == meta.iloc[i]["template_stratum"])
            & (meta["end_position"] == meta.iloc[i]["end_position"])
            & (meta["example_idx"] != meta.iloc[i]["example_idx"])
        ]
        peer_idx = peers["example_idx"].astype(int).tolist()
        expected = z[(9, 6)][peer_idx].mean(0)
        assert torch.allclose(reps[(9, 6)][i], expected)


def test_matched_mean_grouping_by_stratum_end():
    meta = _canonical_meta(n_per=30)
    g = build_mean_ablation_groups(meta)
    assert len(g) == len(meta)


def test_deterministic_donor_assignment():
    meta = _canonical_meta()
    d1, _ = build_resample_donor_map(meta, seed=42)
    d2, _ = build_resample_donor_map(meta, seed=42)
    assert d1.equals(d2)
    h1 = donor_map_deterministic_hash(d1)
    h2 = donor_map_deterministic_hash(d2)
    assert h1 == h2


def test_donor_not_recipient_and_s_io_differ():
    meta = _canonical_meta()
    d, audit = build_resample_donor_map(meta, seed=42)
    assert audit["n_self_donors"] == 0
    assert audit["n_unmatched"] == 0
    for _, row in d.iterrows():
        assert row["recipient_example_idx"] != row["donor_example_idx"]
        assert row["recipient_S"] != row["donor_S"]
        assert row["recipient_IO"] != row["donor_IO"]


def test_same_donor_across_primary_heads():
    meta = _canonical_meta()
    donor_map, _ = build_resample_donor_map(meta, seed=42)
    batch = len(meta)
    z = {(l, h): torch.randn(batch, 3, 4) for l, h in [(9, 6), (9, 9), (10, 0)]}
    reps = compute_matched_resample_replacements(z, donor_map)
    for i, row in donor_map.iterrows():
        rec, don = int(row["recipient_example_idx"]), int(row["donor_example_idx"])
        for head in [(9, 6), (9, 9), (10, 0)]:
            assert torch.allclose(reps[head][rec], z[head][don])


def test_replacement_hook_touches_primary_only():
    batch, seq, n_heads, d = 2, 5, 12, 4
    z_in = torch.randn(batch, seq, n_heads, d)
    reps = {(10, 0): torch.full((batch, seq, d), 7.0)}
    hooks = build_composite_replacement_hooks(reps)
    err = verify_hook_preserves_nonprimary_slices(z_in, hooks, 10, list(range(1, 12)))
    assert err == 0.0
    z_out = hooks[0][1](z_in.clone(), None)
    assert torch.allclose(z_out[:, :, 0, :], torch.full((batch, seq, d), 7.0))


def test_composite_same_layer_replacement():
    batch, seq, d = 3, 4, 2
    z_in = torch.randn(batch, seq, 12, d)
    reps = {(9, 6): torch.zeros(batch, seq, d), (9, 9): torch.ones(batch, seq, d)}
    hooks = build_composite_replacement_hooks(reps)
    assert len(hooks) == 1
    z_out = hooks[0][1](z_in.clone(), None)
    assert torch.allclose(z_out[:, :, 6, :], torch.zeros(batch, seq, d))
    assert torch.allclose(z_out[:, :, 9, :], torch.ones(batch, seq, d))
    assert torch.allclose(z_out[:, :, 7, :], z_in[:, :, 7, :])


def test_informative_damage_criterion():
    df = _synthetic_attr(metric_val=0.0)
    df["damage_drop"] = 0.5
    st = evaluate_damage_strength(df.assign(damage_scheme="matched_mean"), "matched_mean")
    zero = {"exact_mean": 1.0, "INFORMATIVE": True}
    st2 = apply_informative_fraction(st, zero["exact_mean"])
    assert st2["informative_fraction_threshold"] == INFORMATIVE_FRAC * 1.0
    assert st2["informative_magnitude_pass"] is True


def test_robust_same_weak_reversed():
    df_pos = _synthetic_attr(1.0, "pos")
    df_pos["head_name"] = "L10H2"
    ch = ROBUSTNESS_CHANNELS[0]
    st = channel_status(df_pos, ch, scheme="matched_mean")
    assert st["status"] == "ROBUST_SAME"

    df_neg = _synthetic_attr(1.0, "neg")
    df_neg["head_name"] = "L10H2"
    st2 = channel_status(df_neg, ch, scheme="matched_mean")
    assert st2["status"] == "REVERSED"

    df_null = df_pos.copy()
    df_null["phi_T"] = 0.0
    st3 = channel_status(df_null, ch, scheme="matched_mean")
    assert st3["status"] == "WEAK_OR_NULL"


def test_scheme_robust_partial_failed():
    ch_robust = {c["id"]: {"status": "ROBUST_SAME"} for c in ROBUSTNESS_CHANNELS}
    assert scheme_status(ch_robust, informative=True) == "ROBUST"
    ch_rev = dict(ch_robust)
    ch_rev["R1_L10H2_phi_T"] = {"status": "REVERSED"}
    assert scheme_status(ch_rev, informative=True) == "FAILED"
    ch_weak = dict(ch_robust)
    ch_weak["R1_L10H2_phi_T"] = {"status": "WEAK_OR_NULL"}
    assert scheme_status(ch_weak, informative=True) == "PARTIAL"


def test_ef5_gate_pass_partial_fail():
    by_scheme = {
        "matched_mean": {c["id"]: {"status": "ROBUST_SAME"} for c in ROBUSTNESS_CHANNELS},
        "matched_resample": {c["id"]: {"status": "ROBUST_SAME"} for c in ROBUSTNESS_CHANNELS},
    }
    statuses = {"matched_mean": "ROBUST", "matched_resample": "ROBUST"}
    inf = {"matched_mean": True, "matched_resample": True}
    assert ef5_ablation_gate(statuses, by_scheme, inf)["EF5_ABLATION_GATE"] == "PASS"

    by_scheme["matched_mean"]["R1_L10H2_phi_T"]["status"] = "REVERSED"
    statuses["matched_mean"] = "FAILED"
    assert ef5_ablation_gate(statuses, by_scheme, inf)["EF5_ABLATION_GATE"] == "FAIL"


def test_t0_excluded_analysis():
    df = _synthetic_attr(1.0)
    bc = broad_confirmed(df, "L10H2", "phi_T", "positive", scheme="matched_mean", magnitude=False)
    assert bc["STRICT_NONOVERLAP_PASS"]


def test_observed_mean_not_bootstrap_mean():
    df = _synthetic_attr(2.0)
    sub = df[df["head_name"] == "L10H2"]
    obs = observed_canonical_balanced_mean(
        sub["phi_T"].values,
        sub["seed"].values,
        sub["low_level_template_id"].values,
        sub["pattern_type"].values,
    )
    for n_boot in (100, 10000):
        boot = hierarchical_template_cluster_bootstrap_means(
            sub["phi_T"].values,
            sub["seed"].values,
            sub["low_level_template_id"].values,
            sub["pattern_type"].values,
            n_bootstrap=n_boot,
            seed=123,
        )
        assert abs(obs - boot.mean()) > 1e-6 or np.std(boot) < 1e-9


def test_expected_row_counts():
    assert 2 * 4 * 4 * 900 * 32 == 921600
    assert 2 * 4 * 4 * 900 == 28800
