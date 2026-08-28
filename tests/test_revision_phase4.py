"""Tests for Phase-4A canonical IOI expansion."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from revision_standard_ioi import (  # noqa: E402
    ABBA_TEMPLATES,
    BABA_TEMPLATES,
    NAMES_DEDUPED,
    OBJECTS,
    PLACES,
    StandardIOIDataset,
    fill_template,
    legacy_template_overlap_report,
    make_abba_template,
    stratum_name,
    template_inventory_df,
)
from revision_stats import (  # noqa: E402
    hierarchical_template_cluster_bootstrap_means,
    hierarchical_template_cluster_bootstrap_means_buggy_shared_ex_idx,
    stable_sha256_seed,
    template_cluster_bootstrap_indices,
    within_template_stratified_bootstrap_indices,
)


def test_exactly_15_canonical_templates():
    assert len(BABA_TEMPLATES) == 15
    assert len(ABBA_TEMPLATES) == 15
    assert len(set(BABA_TEMPLATES)) == 15


def test_abba_transform_matches_easy_transformer_logic():
    # Spot-check first template
    baba = BABA_TEMPLATES[0]
    abba = make_abba_template(baba)
    assert abba == "Then, [A] and [B] went to the [PLACE]. [B] gave a [OBJECT] to [A]"
    assert ABBA_TEMPLATES[0] == abba
    # Second-clause [B] must remain [B]
    assert abba.count("[A]") == 2
    assert abba.count("[B]") == 2


def test_fill_and_stratum_names():
    s = fill_template(BABA_TEMPLATES[0], A="Alice", B="Bob", place="store", obj="ring")
    assert "Alice" in s and "Bob" in s
    assert stratum_name(3, "BABA") == "T03_BABA"


def test_template_inventory_30_rows():
    inv = template_inventory_df()
    assert len(inv) == 30
    assert set(inv["pattern_type"]) == {"BABA", "ABBA"}


def test_legacy_overlap_exact_only():
    text = legacy_template_overlap_report()
    assert "Legacy B" in text
    assert "EXACT match to BABA template_id=0" in text
    assert "similar" not in text.lower() or "No subjective" in text


def test_stable_sha256_seed_deterministic():
    a = stable_sha256_seed("phase4", 42, "L10H2")
    b = stable_sha256_seed("phase4", 42, "L10H2")
    c = stable_sha256_seed("phase4", 42, "L10H10")
    assert a == b
    assert a != c


def test_within_template_bootstrap_preserves_strata():
    strata = np.array([f"T{t:02d}_{p}" for t in range(15) for p in ("BABA", "ABBA") for _ in range(2)])
    assert len(strata) == 60
    idx = within_template_stratified_bootstrap_indices(strata, n_bootstrap=20, seed=123)
    assert idx.shape == (20, 60)
    for b in range(20):
        drawn = strata[idx[b]]
        for u in np.unique(strata):
            assert np.sum(drawn == u) == np.sum(strata == u)


def test_template_cluster_bootstrap_pairs_baba_abba():
    """Whenever template T is sampled as a cluster, BOTH T_BABA and T_ABBA are included
    with matching within-stratum sample counts (not merely equal global BABA/ABBA totals)."""
    tid = np.array([t for t in range(15) for _p in ("BABA", "ABBA") for _ in range(2)])
    pattern = np.array([p for _t in range(15) for p in ("BABA", "ABBA") for _ in range(2)])
    assert len(tid) == 60
    seed = stable_sha256_seed("cluster", 42)
    idx1 = template_cluster_bootstrap_indices(tid, pattern, n_bootstrap=5, seed=seed)
    idx2 = template_cluster_bootstrap_indices(tid, pattern, n_bootstrap=5, seed=seed)
    assert np.array_equal(idx1, idx2)

    c_baba = {t: int(np.sum((tid == t) & (pattern == "BABA"))) for t in range(15)}
    c_abba = {t: int(np.sum((tid == t) & (pattern == "ABBA"))) for t in range(15)}
    assert all(c_baba[t] == 2 and c_abba[t] == 2 for t in range(15))

    for b in range(5):
        drawn_tid = tid[idx1[b]]
        drawn_pat = pattern[idx1[b]]
        assert len(drawn_tid) == 60
        assert np.sum(drawn_pat == "BABA") == 30
        assert np.sum(drawn_pat == "ABBA") == 30

        k_sum = 0
        for t in range(15):
            n_baba = int(np.sum((drawn_tid == t) & (drawn_pat == "BABA")))
            n_abba = int(np.sum((drawn_tid == t) & (drawn_pat == "ABBA")))
            assert n_baba % c_baba[t] == 0
            assert n_abba % c_abba[t] == 0
            k_baba = n_baba // c_baba[t]
            k_abba = n_abba // c_abba[t]
            # Paired: same cluster multiplicity for BABA and ABBA of T
            assert k_baba == k_abba, (
                f"draw {b} template {t}: BABA multiplicity {k_baba} != ABBA {k_abba}"
            )
            k_sum += k_baba
        assert k_sum == 15, f"draw {b}: expected 15 cluster slots, got {k_sum}"


def test_no_historical_overwrite_targets_exist_or_none():
    # Phase-4 must not write into these prefixes; existence is fine.
    forbidden_writes = [
        REPO / "results/raw/fali_decomposition_raw.csv",
        REPO / "results/revision_phase1/full/seed_42/candidate_raw.csv",
    ]
    # Just ensure we can hash them / they are not required to be absent
    for p in forbidden_writes:
        if p.exists():
            assert p.is_file()


@pytest.fixture(scope="module")
def gpt2_cpu():
    pytest.importorskip("transformer_lens")
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        center_writing_weights=True,
        device="cpu",
        dtype="float32",
    )
    model.eval()
    return model


def test_balanced_deterministic_dataset(gpt2_cpu):
    d1 = StandardIOIDataset(gpt2_cpu, num_per_stratum=2, seed=42)
    d2 = StandardIOIDataset(gpt2_cpu, num_per_stratum=2, seed=42)
    d3 = StandardIOIDataset(gpt2_cpu, num_per_stratum=2, seed=1)
    assert d1.num_samples == 60
    assert d1.metadata_df["template_stratum"].nunique() == 30
    assert (d1.metadata_df.groupby("template_stratum").size() == 2).all()
    assert d1.metadata_df["low_level_template_id"].nunique() == 15
    assert set(d1.metadata_df["pattern_type"]) == {"BABA", "ABBA"}
    assert d1.metadata_df["prompt"].tolist() == d2.metadata_df["prompt"].tolist()
    assert d1.metadata_df["prompt"].tolist() != d3.metadata_df["prompt"].tolist()
    # serialization
    cols = [
        "example_idx",
        "low_level_template_id",
        "pattern_type",
        "template_stratum",
        "prompt",
        "S",
        "IO",
        "S1_position",
        "S2_position",
        "IO_position",
        "end_position",
        "place",
        "object",
        "seed",
    ]
    for c in cols:
        assert c in d1.metadata_df.columns
    filt = d1.tokenization_filter_report()
    assert filt["usable_names"] >= 90
    assert len(filt["rejected_names"]) == 0


def test_token_positions_variable_length_and_padding(gpt2_cpu):
    d = StandardIOIDataset(gpt2_cpu, num_per_stratum=1, seed=42)
    assert d.num_samples == 30
    eos = d.tokenizer.eos_token_id
    lengths = []
    for i in range(d.batch_size):
        end = int(d.end[i].item())
        lengths.append(end + 1)
        assert int(d.clean_tokens[i, end].item()) != eos
        # padding after end is eos when shorter than max
        if end + 1 < d.seq_len:
            assert int(d.clean_tokens[i, end + 1].item()) == eos
        s_id = int(d.s_token_ids[i].item())
        io_id = int(d.io_token_ids[i].item())
        toks = d.clean_tokens[i, : end + 1]
        assert (toks == s_id).sum().item() == 2
        assert (toks == io_id).sum().item() == 1
        # positions match
        assert int(d.S1[i].item()) == int((toks == s_id).nonzero()[0].item())
        s_pos = (toks == s_id).nonzero().view(-1).tolist()
        assert s_pos == [int(d.S1[i].item()), int(d.S2[i].item())]
        assert int(d.IO[i].item()) == int((toks == io_id).nonzero()[0].item())
    # Variable lengths across templates
    assert len(set(lengths)) > 1


def test_single_token_name_validation_explicit(gpt2_cpu):
    # Force a multi-token fake name into the filter path
    from revision_standard_ioi import filter_single_token_words

    usable, rejected = filter_single_token_words(
        gpt2_cpu.tokenizer, ["Michael", "XxNotARealNameQqq"]
    )
    assert "Michael" in usable
    assert any(r["word"] == "XxNotARealNameQqq" for r in rejected)


def _synthetic_balanced_panel(n_templates: int = 4, n_per: int = 5, seeds=(42, 1, 8, 22)):
    """Deterministic synthetic panel with unique values per seed×cell×example."""
    rows = []
    for s in seeds:
        for t in range(n_templates):
            for p in ("BABA", "ABBA"):
                for j in range(n_per):
                    # Unique encoding so resampling choices are observable in means
                    val = float(s) * 1e6 + t * 1e3 + (0 if p == "BABA" else 500) + j
                    rows.append((s, t, p, val))
    seed_ids = np.array([r[0] for r in rows])
    tid = np.array([r[1] for r in rows])
    pattern = np.array([r[2] for r in rows])
    values = np.array([r[3] for r in rows], dtype=np.float64)
    return values, seed_ids, tid, pattern


def test_hierarchical_bootstrap_independent_per_seed_resampling():
    """
    Preregistered hierarchical procedure:
      - shared template clusters across seeds within a draw;
      - BABA/ABBA paired for each selected template;
      - independent within-cell example draws per seed;
      - equal weighting of seed means;
      - SHA256-stable determinism.

    The pre-fix buggy implementation reuses one ex_idx across seeds and must
    fail the independence assertion.
    """
    values, seed_ids, tid, pattern = _synthetic_balanced_panel()
    n_templates = 4
    n_boot = 64
    rng_seed = stable_sha256_seed("phase4c", "hier_test", 7)

    means_a, drawn_t, ex_idx = hierarchical_template_cluster_bootstrap_means(
        values,
        seed_ids,
        tid,
        pattern,
        n_bootstrap=n_boot,
        seed=rng_seed,
        n_templates=n_templates,
        return_draw_arrays=True,
    )
    means_b, drawn_t2, ex_idx2 = hierarchical_template_cluster_bootstrap_means(
        values,
        seed_ids,
        tid,
        pattern,
        n_bootstrap=n_boot,
        seed=rng_seed,
        n_templates=n_templates,
        return_draw_arrays=True,
    )
    # Determinism
    assert np.array_equal(means_a, means_b)
    assert np.array_equal(drawn_t, drawn_t2)
    assert np.array_equal(ex_idx, ex_idx2)

    n_seeds = len(np.unique(seed_ids))
    assert drawn_t.shape == (n_boot, n_templates)
    assert ex_idx.shape == (n_boot, n_seeds, n_templates, 2, 5)

    # Shared template clusters: single drawn_t array (not per-seed)
    # Pairing: for every slot, both pattern indices 0 and 1 are always used (shape dim)
    assert ex_idx.shape[3] == 2

    # Independent example resampling across seeds
    assert not np.array_equal(ex_idx[:, 0], ex_idx[:, 1]), (
        "ex_idx must differ independently across seeds"
    )
    # At least one bootstrap draw has different indices for seed 0 vs 1
    diffs = [not np.array_equal(ex_idx[b, 0], ex_idx[b, 1]) for b in range(n_boot)]
    assert any(diffs)

    # Equal seed weighting: if each seed is a constant offset, grand mean = avg of seeds
    # Construct constants per seed and check one draw analytically via means function
    const_vals = seed_ids.astype(np.float64)  # value == seed id
    m = hierarchical_template_cluster_bootstrap_means(
        const_vals,
        seed_ids,
        tid,
        pattern,
        n_bootstrap=20,
        seed=rng_seed,
        n_templates=n_templates,
    )
    expected = float(np.mean(sorted(np.unique(seed_ids).astype(np.float64))))
    assert np.allclose(m, expected)

    # Buggy shared-ex_idx implementation: shape lacks seed axis → independence fails
    _, _, buggy_ex = hierarchical_template_cluster_bootstrap_means_buggy_shared_ex_idx(
        values,
        seed_ids,
        tid,
        pattern,
        n_bootstrap=n_boot,
        seed=rng_seed,
        n_templates=n_templates,
        return_draw_arrays=True,
    )
    assert buggy_ex.ndim == 4  # (B, T, P, E) — no seed axis
    # The fixed test criterion (ex_idx[:,0] != ex_idx[:,1]) cannot even be stated;
    # simulating reuse: identical indices applied to all seeds.
    assert buggy_ex.shape[0] == n_boot
    with pytest.raises(AssertionError):
        # Mimic the independence check against a broadcast of buggy_ex
        fake_shared = np.stack([buggy_ex] * n_seeds, axis=1)
        assert not np.array_equal(fake_shared[:, 0], fake_shared[:, 1])

