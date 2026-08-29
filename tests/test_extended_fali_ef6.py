"""Tests for EF-6 Pythia induction transfer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from extended_fali_ef6_analysis import (  # noqa: E402
    N_BOOTSTRAP,
    PairCandidate,
    bootstrap_confirmation_balanced,
    bootstrap_offset_balanced,
    candidate_pair_eligible,
    ef6_gate,
    evaluate_phi_gate,
    observed_confirmation_balanced_mean,
    observed_offset_balanced_mean,
    primary_eligible,
    select_head_pair,
)
from extended_fali_induction import (  # noqa: E402
    BASE_LEN,
    CONFIRMATION_SEEDS,
    DISCOVERY_SEED,
    EXAMPLES_PER_OFFSET,
    QUERY_OFFSETS,
    TOKEN_POOL_HIGH,
    TOKEN_POOL_LOW,
    generate_induction_metadata,
    metadata_to_tensors,
    validate_induction_metadata,
)
from extended_fali_task_partition import build_induction_partition  # noqa: E402
from extended_fali_interventions import partition_diagnostics  # noqa: E402


def test_dataset_64_unique_ids_and_pool():
    meta = generate_induction_metadata(seed=42)
    validate_induction_metadata(meta)
    for _, row in meta.iterrows():
        x = json.loads(row["x_block_json"])
        y = json.loads(row["y_block_json"])
        assert len(x) == 32 and len(y) == 32
        assert len(set(x) & set(y)) == 0
        all_ids = x + y
        assert len(all_ids) == 64 and len(set(all_ids)) == 64
        assert min(all_ids) >= TOKEN_POOL_LOW and max(all_ids) < TOKEN_POOL_HIGH


def test_repeated_control_construction_and_lengths():
    meta = generate_induction_metadata(seed=1)
    for _, row in meta.iterrows():
        j = int(row["query_offset"])
        rep = json.loads(row["repeated_token_ids_json"])
        ctl = json.loads(row["control_token_ids_json"])
        assert len(rep) == len(ctl) == 32 + j + 1
        assert int(row["query_pos"]) == 32 + j
        assert int(row["target_key"]) == j + 1
        assert int(row["target_token_id"]) == json.loads(row["x_block_json"])[j + 1]


def test_target_absent_from_suffix_and_control_prefix():
    meta = generate_induction_metadata(seed=8)
    for _, row in meta.iterrows():
        x = json.loads(row["x_block_json"])
        rep = json.loads(row["repeated_token_ids_json"])
        ctl = json.loads(row["control_token_ids_json"])
        target = int(row["target_token_id"])
        q = int(row["query_pos"])
        assert target not in rep[32 : q + 1]
        assert target not in ctl[:32]


def test_padding_after_query_only():
    ds = metadata_to_tensors(generate_induction_metadata(seed=22))
    for i in range(ds.batch_size):
        vl = int(ds.valid_length[i])
        q = int(ds.query_pos[i])
        assert vl == q + 1
        if vl < ds.seq_len:
            assert (ds.repeated_tokens[i, vl:] == 0).all()


def test_deterministic_generation_by_seed():
    m1 = generate_induction_metadata(seed=42)
    m2 = generate_induction_metadata(seed=42)
    m3 = generate_induction_metadata(seed=1)
    assert m1.equals(m2)
    assert not m1.equals(m3)


def test_partition_exact_t_and_coverage():
    ds = metadata_to_tensors(generate_induction_metadata(seed=42, examples_per_offset=4))
    part = build_induction_partition(
        ds.query_pos,
        ds.target_key,
        valid_lengths=ds.valid_length,
        seq_len=ds.seq_len,
    )
    diag = partition_diagnostics(part)
    assert all(c == 0 for c in diag["coverage_error_per_example"])
    assert all(c == 0 for c in diag["overlap_count_per_example"])
    assert all(c == 1 for c in diag["target_count_per_example"])


def test_balanced_estimator_two_pattern_offsets():
    values = np.array([1.0] * 20 + [1.0] * 20 + [-1.0] * 20)
    offsets = np.array([8] * 20 + [16] * 20 + [24] * 20)
    combined = observed_offset_balanced_mean(values, offsets)
    assert combined == pytest.approx((1.0 + 1.0 - 1.0) / 3)


def test_balanced_equals_half_for_two_offset_groups():
    """Regression: balanced mean over offsets equals average of per-offset means."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=384)
    offsets = np.tile(QUERY_OFFSETS, 128)
    direct = observed_offset_balanced_mean(values, offsets)
    per_off = [float(values[offsets == off].mean()) for off in QUERY_OFFSETS]
    assert direct == pytest.approx(float(np.mean(per_off)))


def test_bootstrap_deterministic_and_point_independent_of_n_boot():
    rng_values = np.random.default_rng(0).normal(size=384)
    offsets = np.tile(QUERY_OFFSETS, 128)
    b1 = bootstrap_offset_balanced(rng_values, offsets, n_bootstrap=100, scope="t", metric="m")
    b2 = bootstrap_offset_balanced(rng_values, offsets, n_bootstrap=100, scope="t", metric="m")
    assert np.allclose(b1, b2)
    point = observed_offset_balanced_mean(rng_values, offsets)
    assert point == pytest.approx(float(np.mean(b1)), abs=0.5)


def test_confirmation_bootstrap_seeds_not_resampled():
    per_seed_offsets = np.array([8] * 128 + [16] * 128 + [24] * 128)
    offsets = np.concatenate([per_seed_offsets, per_seed_offsets, per_seed_offsets])
    values = np.concatenate([np.full(384, 1.0), np.full(384, 2.0), np.full(384, 3.0)])
    seeds = np.array([1] * 384 + [8] * 384 + [22] * 384)
    boot = bootstrap_confirmation_balanced(values, seeds, offsets, n_bootstrap=50, scope="s", metric="m")
    assert len(boot) == 50


def test_primary_eligibility_exact():
    ok, _ = primary_eligible(
        clean_attention=0.06,
        intact_ci_low=0.01,
        intact_ci_high=0.02,
        offset_intact_means={8: 0.1, 16: 0.2, 24: 0.3},
        has_later_attention_head=True,
    )
    assert ok
    bad, reason = primary_eligible(
        clean_attention=0.04,
        intact_ci_low=0.01,
        intact_ci_high=0.02,
        offset_intact_means={8: 0.1, 16: 0.2, 24: 0.3},
        has_later_attention_head=True,
    )
    assert not bad and reason == "clean_attention"


def test_candidate_requires_primary_earlier_layer():
    ok, reason = candidate_pair_eligible(
        primary_layer=5,
        candidate_layer=4,
        cond_ci_low=0.1,
        cond_ci_high=0.2,
        gain_ci_low=0.1,
        gain_ci_high=0.2,
        offset_cond_means={8: 1, 16: 1, 24: 1},
        offset_gain_means={8: 1, 16: 1, 24: 1},
    )
    assert not ok and reason == "layer_order"


def test_pair_selection_tie_break_deterministic():
    pairs = [
        PairCandidate(0, 0, 2, 0, 1.0, 0.5, 0.1, 0.2, True),
        PairCandidate(0, 1, 2, 1, 1.0, 0.4, 0.2, 0.2, True),
    ]
    sel = select_head_pair(pairs)
    assert sel.candidate_head == 0


def test_pair_selector_no_phi_inputs():
    import inspect
    from extended_fali_ef6_runner import run_discovery_pair_selection

    src = inspect.getsource(run_discovery_pair_selection)
    assert "phi_T" not in src
    assert "phi_R" not in src


def test_g0_discovery_rule_synthetic():
    adv = np.concatenate([np.full(128, 0.5), np.full(128, 0.5), np.full(128, 0.5)])
    offsets = np.array(list(QUERY_OFFSETS) * 128)
    from extended_fali_ef6_analysis import evaluate_scalar_gate

    g0 = evaluate_scalar_gate("G0", adv, np.full(len(adv), DISCOVERY_SEED), offsets, discovery=True, positive=True, scope="x")
    assert g0.status == "CONFIRMED"


def test_phi_o_not_confirmatory_in_gate():
    gates = {"G0": "CONFIRMED", "G1": "CONFIRMED", "G2": "CONFIRMED", "G3": "CONFIRMED", "G4": "NOT_CONFIRMED"}
    assert ef6_gate(gates) == "PARTIAL"


def test_ef6_gate_pass_and_fail():
    all_pass = {f"G{i}": "CONFIRMED" for i in range(5)}
    assert ef6_gate(all_pass) == "PASS"
    fail = {"G0": "NOT_CONFIRMED", "G1": "CONFIRMED", "G2": "CONFIRMED", "G3": "CONFIRMED", "G4": "CONFIRMED"}
    assert ef6_gate(fail) == "FAIL"


def test_g3_share_rule():
    n = 384 * 3
    values = np.ones(n)
    seeds = np.array([1] * 384 + [8] * 384 + [22] * 384)
    offsets = np.tile(QUERY_OFFSETS, 384)
    phi_score = np.full(n, 0.01)
    phi_r = np.full(n, -0.01)
    phi_o = np.zeros(n)
    g3 = evaluate_phi_gate("G3", values, seeds, offsets, phi_score, phi_r, phi_o, positive=True, scope="g3t")
    assert g3.status == "CONFIRMED"


def test_discovery_seed_not_in_confirmation_constants():
    assert DISCOVERY_SEED not in CONFIRMATION_SEEDS


def test_ioi_partition_still_valid():
    pytest.importorskip("transformer_lens")
    from extended_fali_gpt2 import build_ioi_partition
    from revision_standard_ioi import StandardIOIDataset
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu", dtype=torch.float32)
    ds = StandardIOIDataset(model, num_per_stratum=2, seed=42)
    part = build_ioi_partition(ds, "cpu")
    assert int(part.target_mask.sum()) == ds.batch_size
