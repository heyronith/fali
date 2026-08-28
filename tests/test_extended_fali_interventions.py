"""Tests for Extended FALI intervention primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from extended_fali_interventions import (  # noqa: E402
    build_causal_valid_mask,
    build_hybrid_scores,
    build_score_partition_masks,
    enumerate_extended_fali_cells,
    masked_softmax,
    reconstruct_z_from_scores_values,
    validate_score_partition,
)
from extended_fali_synthetic import (  # noqa: E402
    LEVEL_A_CASES,
    build_level_a_tc_interaction,
    single_channel_owen_check,
)


def test_partition_disjoint_coverage():
    part = build_score_partition_masks(6, final_query=5, target_key=3, valid_len=6)
    validate_score_partition(part)
    assert part.valid_mask.sum() == part.target_mask.sum() + part.competing_mask.sum() + part.other_query_mask.sum()


def test_variable_length_padding():
    part = build_score_partition_masks(8, final_query=4, target_key=2, valid_len=5)
    validate_score_partition(part)
    invalid = ~part.valid_mask
    scores = torch.randn(1, 8, 8)
    probs = masked_softmax(scores, part.valid_mask)
    assert float(probs.masked_select(invalid.unsqueeze(0)).max()) <= 1e-8


def test_thirty_two_cells():
    assert len(list(enumerate_extended_fali_cells())) == 32


@pytest.mark.parametrize("name,intended", [(k, v[1]) for k, v in LEVEL_A_CASES.items()])
def test_level_a_single_channel(name: str, intended: str):
    builder, _ = LEVEL_A_CASES[name]
    result = single_channel_owen_check(builder(), intended)
    assert result["intended_err"] < 1e-12
    for key, val in result.items():
        if key.startswith("other_"):
            assert val < 1e-12


def test_level_a_tc_interaction_pipeline():
    system = build_level_a_tc_interaction(2.0)
    table = system.cell_table()
    assert table[(1, 0, 0, 0, 0)] < 1e-9
    assert table[(0, 0, 0, 0, 1)] < 1e-12
    assert table[(1, 0, 0, 0, 1)] > 1.0
    owen = system.owen()
    delta = table[(1, 0, 0, 0, 1)] - table[(0, 0, 0, 0, 0)]
    assert abs(owen["T"] - delta / 2) < 1e-12
    assert abs(owen["C"] - delta / 2) < 1e-12
    assert abs(owen["R"]) < 1e-12


def test_hybrid_reconstruction_all_a_all_b():
    part = build_score_partition_masks(4, 3, 2)
    sa = torch.randn(1, 4, 4, dtype=torch.float64)
    sb = torch.randn(1, 4, 4, dtype=torch.float64)
    va = torch.randn(1, 4, 3, dtype=torch.float64)
    vb = torch.randn(1, 4, 3, dtype=torch.float64)
    hs_a = build_hybrid_scores(sa, sb, part, (0, 0, 0, 0, 0))
    hs_b = build_hybrid_scores(sa, sb, part, (1, 1, 1, 0, 0))
    valid = part.valid_mask
    sa_m = sa.masked_fill(~valid.unsqueeze(0), float("-inf"))
    sb_m = sb.masked_fill(~valid.unsqueeze(0), float("-inf"))
    diff_a = (hs_a - sa_m).masked_fill(~valid.unsqueeze(0), 0.0)
    diff_b = (hs_b - sb_m).masked_fill(~valid.unsqueeze(0), 0.0)
    assert float(diff_a.abs().max()) < 1e-12
    assert float(diff_b.abs().max()) < 1e-12
    pa = masked_softmax(hs_a, valid)
    pb = masked_softmax(hs_b, valid)
    za = reconstruct_z_from_scores_values(pa, va)
    zb = reconstruct_z_from_scores_values(pb, vb)
    assert float((za - reconstruct_z_from_scores_values(masked_softmax(sa_m, valid), va)).abs().max()) < 1e-12


def test_causal_valid_mask_shape():
    m = build_causal_valid_mask(5)
    assert m.shape == (5, 5)
    assert m.tril().sum() == m.sum()
