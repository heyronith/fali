"""
Automated tests for Revision Phase-1 conditional-gain infrastructure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from revision_config import (  # noqa: E402
    BACKUP_NAME_MOVERS,
    EXTRA_CONTROLS,
    NEGATIVE_NAME_MOVERS,
    PRIMARY_NAME_MOVERS,
    all_phase1_candidates,
)
from revision_dataset import template_id_for_index  # noqa: E402
from revision_interventions import (  # noqa: E402
    build_zero_ablation_hooks,
    group_heads_by_layer,
    hook_point_names,
)
from revision_stats import stratified_bootstrap_ci  # noqa: E402


# ---------------------------------------------------------------------------
# Hook tests
# ---------------------------------------------------------------------------


def test_group_heads_by_layer_dedup_and_sort():
    grouped = group_heads_by_layer([(10, 10), (9, 9), (10, 0), (9, 6), (10, 0)])
    assert grouped == {9: [6, 9], 10: [0, 10]}


def test_build_zero_ablation_hooks_one_per_layer():
    heads = [(9, 6), (9, 9), (10, 0), (10, 10)]
    hooks = build_zero_ablation_hooks(heads)
    names = hook_point_names(hooks)
    assert names == ["blocks.9.attn.hook_z", "blocks.10.attn.hook_z"]
    assert len(names) == len(set(names))


def test_composite_hook_zeros_requested_heads_only_and_does_not_mutate_input():
    hooks = build_zero_ablation_hooks([(9, 6), (9, 9)])
    assert len(hooks) == 1
    _, hook_fn = hooks[0]

    z = torch.randn(2, 5, 12, 8)
    z_original = z.clone()
    out = hook_fn(z, hook=None)

    # Input unchanged
    assert torch.equal(z, z_original)
    # Requested heads zeroed
    assert torch.count_nonzero(out[:, :, 6, :]) == 0
    assert torch.count_nonzero(out[:, :, 9, :]) == 0
    # Other heads unchanged
    for h in range(12):
        if h in (6, 9):
            continue
        assert torch.equal(out[:, :, h, :], z_original[:, :, h, :])


def test_multi_layer_hooks_independent():
    hooks = build_zero_ablation_hooks([(9, 6), (10, 0), (11, 2)])
    assert hook_point_names(hooks) == [
        "blocks.9.attn.hook_z",
        "blocks.10.attn.hook_z",
        "blocks.11.attn.hook_z",
    ]


def test_head_bounds_assertion():
    with pytest.raises(AssertionError):
        build_zero_ablation_hooks([(12, 0)])
    with pytest.raises(AssertionError):
        build_zero_ablation_hooks([(0, 12)])


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------


def test_conditional_gain_identity():
    rng = np.random.default_rng(0)
    A = rng.normal(size=20)
    B = rng.normal(size=20)
    D = rng.normal(size=20)
    C = rng.normal(size=20)
    intact = A - D
    conditional = B - C
    gain = conditional - intact
    assert np.allclose(gain, (B - C) - (A - D))
    assert np.allclose(gain, B - C - A + D)


# ---------------------------------------------------------------------------
# Bootstrap tests
# ---------------------------------------------------------------------------


def test_bootstrap_deterministic_and_ordered():
    values = np.array([0.1, -0.2, 0.3, 0.05, -0.1, 0.2, 0.0, 0.15])
    strata = np.array(["A", "B", "A", "B", "A", "B", "A", "B"])
    m1, lo1, hi1 = stratified_bootstrap_ci(values, strata, n_bootstrap=500, seed=123)
    m2, lo2, hi2 = stratified_bootstrap_ci(values, strata, n_bootstrap=500, seed=123)
    assert m1 == m2 and lo1 == lo2 and hi1 == hi2
    assert lo1 <= m1 <= hi1


def test_bootstrap_preserves_stratum_sizes(monkeypatch):
    values = np.arange(10, dtype=np.float64)
    strata = np.array(["A"] * 4 + ["B"] * 6)
    seen_counts = []

    real_rng_cls = np.random.default_rng

    class TrackingRNG:
        def __init__(self, seed):
            self._rng = real_rng_cls(seed)

        def choice(self, idxs, size=None, replace=True):
            seen_counts.append((len(idxs), size))
            return self._rng.choice(idxs, size=size, replace=replace)

    monkeypatch.setattr(np.random, "default_rng", lambda seed: TrackingRNG(seed))
    stratified_bootstrap_ci(values, strata, n_bootstrap=3, seed=0)
    # Each bootstrap draw should sample 4 from A-pool and 6 from B-pool
    assert seen_counts.count((4, 4)) == 3
    assert seen_counts.count((6, 6)) == 3


# ---------------------------------------------------------------------------
# Dataset / taxonomy tests
# ---------------------------------------------------------------------------


def test_template_assignment_mixed():
    assert template_id_for_index(0, "mixed") == "A"
    assert template_id_for_index(1, "mixed") == "B"
    assert template_id_for_index(2, "mixed") == "A"


def test_candidate_count_and_taxonomy_frozen():
    assert PRIMARY_NAME_MOVERS == [(9, 6), (9, 9), (10, 0)]
    assert len(BACKUP_NAME_MOVERS) == 8
    assert len(NEGATIVE_NAME_MOVERS) == 2
    assert len(EXTRA_CONTROLS) == 2
    assert len(all_phase1_candidates()) == 12


def test_dataset_positions_with_model():
    pytest.importorskip("transformer_lens")
    from transformer_lens import HookedTransformer
    from revision_dataset import make_validated_ioi_dataset, build_dataset_metadata

    model = HookedTransformer.from_pretrained(
        "gpt2-small", center_writing_weights=True, device="cpu"
    )
    dataset = make_validated_ioi_dataset(model, num_samples=4, seed=42, template_style="mixed")
    meta = build_dataset_metadata(dataset)
    assert list(meta["template_id"]) == ["A", "B", "A", "B"]
    for i in range(4):
        toks = dataset.clean_tokens[i]
        end = int(dataset.end[i])
        assert toks[end].item() == model.tokenizer.encode(" to")[0]
        assert toks[end].item() != model.tokenizer.eos_token_id


# ---------------------------------------------------------------------------
# Output path / smoke output tests
# ---------------------------------------------------------------------------


def test_output_dir_rejects_historical_paths():
    import importlib.util

    exp_path = REPO_ROOT / "experiments" / "revision_phase1_conditional_gain.py"
    spec = importlib.util.spec_from_file_location(
        "revision_phase1_conditional_gain", exp_path
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    with pytest.raises(ValueError):
        mod.assert_safe_output_dir((REPO_ROOT / "results/raw").resolve())
    with pytest.raises(ValueError):
        mod.assert_safe_output_dir((REPO_ROOT / "results/summary").resolve())
    # Allowed
    mod.assert_safe_output_dir((REPO_ROOT / "results/revision_phase1/smoke").resolve())


def test_smoke_outputs_exist_if_present():
    """If a prior smoke run exists, check required files. Otherwise skip."""
    out = REPO_ROOT / "results/revision_phase1/smoke"
    required = [
        "run_config.json",
        "environment.json",
        "dataset_metadata.csv",
        "primary_ablation_summary.csv",
        "candidate_summary.csv",
        "candidate_raw.csv",
        "bootstrap_summary.csv",
        "correlation_summary.csv",
        "run_log.txt",
        "validation_report.md",
    ]
    if not out.exists():
        pytest.skip("Smoke outputs not yet generated")
    missing = [r for r in required if not (out / r).exists()]
    assert not missing, f"Missing smoke outputs: {missing}"
    with open(out / "run_config.json") as f:
        cfg = json.load(f)
    assert cfg["num_samples"] == 8
    assert len(cfg["candidates"]) == 12
