"""Structural tests for EF-2 GPT-2 smoke (no model download required)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments"))

from extended_fali_gpt2 import (  # noqa: E402
    QUOTIENT_STATE_MAP,
    align_dataset_to_reference,
    build_ioi_partition,
    historical_quotient_cells,
    owen_diagnostics,
)
from extended_fali_interventions import enumerate_extended_fali_cells  # noqa: E402
from revision_config import all_phase1_candidates  # noqa: E402


class _MockDataset:
    def __init__(self, ends, ios, seq_len=16, batch=None):
        self.batch_size = batch or len(ends)
        self.end = torch.tensor(ends, dtype=torch.long)
        self.IO = torch.tensor(ios, dtype=torch.long)
        self.clean_tokens = torch.zeros(self.batch_size, seq_len, dtype=torch.long)
        self.io_token_ids = torch.ones(self.batch_size, dtype=torch.long)
        self.s_token_ids = torch.zeros(self.batch_size, dtype=torch.long)


def test_ioi_partition_from_task_positions():
    ds = _MockDataset(ends=[14, 15], ios=[4, 5], seq_len=20)
    part = build_ioi_partition(ds, "cpu")
    assert part.target_mask[0, 14, 4]
    assert part.target_mask[1, 15, 5]
    assert int(part.target_mask[0].sum()) == 1
    assert int(part.target_mask[1].sum()) == 1


def test_quotient_state_map_covers_eight_cells():
    assert len(QUOTIENT_STATE_MAP) == 8
    states = set(QUOTIENT_STATE_MAP.values())
    assert len(states) == 8
    for s in states:
        t, r, o, v, c = s
        assert t == r == o


def test_cell_enumeration_count():
    assert len(list(enumerate_extended_fali_cells())) == 32


def test_candidate_head_count():
    assert len(all_phase1_candidates()) == 12


def test_expected_ef2_row_counts():
    assert 12 * 8 * 32 == 3072
    assert 12 * 8 == 96


def test_align_dataset_passes_on_reference():
    ref = pd.read_csv(REPO / "results/revision_phase3c/smoke/dataset_metadata.csv")
    status, lines = align_dataset_to_reference(ref, ref)
    assert status == "PASS"
    assert any("PASS" in ln for ln in lines)


def test_owen_efficiency_on_synthetic_table():
    table = {s: float(i) for i, (_, s) in enumerate(enumerate_extended_fali_cells())}
    diag = owen_diagnostics(table)
    assert diag["owen_residual"] <= 1e-6


def test_historical_quotient_keys():
    table = {(0, 0, 0, 0, 0): 0.0, (1, 1, 1, 1, 1): 4.0}
    for _, s in enumerate_extended_fali_cells():
        if s not in table:
            table[s] = 1.0
    h = historical_quotient_cells(table)
    assert set(h.keys()) == {
        "H000",
        "H100",
        "H010",
        "H110",
        "H001",
        "H101",
        "H011",
        "H111",
    }


def test_ef2_output_dir_guard():
    from extended_fali_ef2_gpt2_smoke import assert_safe_output_dir

    assert_safe_output_dir(REPO / "results/extended_fali/ef2_gpt2_smoke")
    with pytest.raises(ValueError):
        assert_safe_output_dir(REPO / "results/revision_phase3c/smoke")
