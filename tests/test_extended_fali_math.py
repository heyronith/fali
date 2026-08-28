"""Synthetic algebra tests for Extended FALI standard Owen mathematics."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from src.extended_fali_math import (
    EXTENDED_FALI_GROUPS,
    EXTENDED_FALI_PLAYERS,
    N_EXTENDED_FALI_ADMISSIBLE_PERMUTATIONS,
    N_EXTENDED_FALI_CELLS,
    additive_char_fn,
    admissible_owen_permutations,
    block_interaction_char_fn,
    compare_owen_implementations,
    constant_char_fn,
    extended_fali_quotient_game,
    extended_fali_quotient_shapley,
    interaction_char_fn,
    owen_block_order_invariance_max,
    owen_value_permutation,
    owen_value_subset,
    owen_within_block_order_invariance_max,
    random_char_fn,
    shapley_attention_two_player,
    two_player_attention_char_fn,
    verify_efficiency,
    verify_quotient_consistency,
)
from src.revision_fali import fali_shapley


PLAYERS = EXTENDED_FALI_PLAYERS
GROUPS = EXTENDED_FALI_GROUPS
ATOL = 1e-12


def _owen(char_fn):
    subset = owen_value_subset(PLAYERS, GROUPS, char_fn)
    perm = owen_value_permutation(PLAYERS, GROUPS, char_fn)
    return subset, perm


def test_admissible_permutation_count_and_uniqueness():
    perms = admissible_owen_permutations(GROUPS)
    assert len(perms) == 36
    assert len(set(perms)) == 36
    assert N_EXTENDED_FALI_ADMISSIBLE_PERMUTATIONS == 36


def test_constant_game_zero_attribution():
    char_fn = constant_char_fn(PLAYERS, 1.0)
    phi, perm = _owen(char_fn)
    for p in PLAYERS:
        assert abs(phi[p]) < ATOL
        assert abs(perm[p]) < ATOL
    assert verify_efficiency(PLAYERS, char_fn, phi) < ATOL


@pytest.mark.parametrize("player", PLAYERS)
def test_additive_single_factor(player: str):
    char_fn = additive_char_fn(PLAYERS, {player: 1.0})
    phi, perm = _owen(char_fn)
    for p in PLAYERS:
        expected = 1.0 if p == player else 0.0
        assert abs(phi[p] - expected) < ATOL
        assert abs(perm[p] - expected) < ATOL
    assert verify_efficiency(PLAYERS, char_fn, phi) < ATOL


def test_additive_five_factor_exact_recovery():
    coeffs = {"T": 0.3, "R": -0.7, "O": 0.2, "V": 1.1, "C": -0.4}
    char_fn = additive_char_fn(PLAYERS, coeffs)
    phi, perm = _owen(char_fn)
    for p in PLAYERS:
        assert abs(phi[p] - coeffs[p]) < ATOL
        assert abs(perm[p] - coeffs[p]) < ATOL
    assert verify_efficiency(PLAYERS, char_fn, phi) < ATOL


def test_tr_pure_interaction():
    char_fn = interaction_char_fn(PLAYERS, frozenset({"T", "R"}))
    phi, perm = _owen(char_fn)
    for impl in (phi, perm):
        assert abs(impl["T"] - 0.5) < ATOL
        assert abs(impl["R"] - 0.5) < ATOL
        assert abs(impl["O"]) < ATOL
        assert abs(impl["V"]) < ATOL
        assert abs(impl["C"]) < ATOL


def test_tc_pure_interaction():
    char_fn = interaction_char_fn(PLAYERS, frozenset({"T", "C"}))
    phi, perm = _owen(char_fn)
    for impl in (phi, perm):
        assert abs(impl["T"] - 0.5) < ATOL
        assert abs(impl["C"] - 0.5) < ATOL
        assert abs(impl["R"]) < ATOL
        assert abs(impl["O"]) < ATOL
        assert abs(impl["V"]) < ATOL


def test_trc_pure_interaction():
    char_fn = interaction_char_fn(PLAYERS, frozenset({"T", "R", "C"}))
    phi, perm = _owen(char_fn)
    for impl in (phi, perm):
        assert abs(impl["T"] - 0.25) < ATOL
        assert abs(impl["R"] - 0.25) < ATOL
        assert abs(impl["C"] - 0.5) < ATOL
        assert abs(impl["O"]) < ATOL
        assert abs(impl["V"]) < ATOL


@pytest.mark.parametrize(
    "block_indices,expected_nonzero",
    [
        ((0, 1), {"T", "R", "O", "V"}),
        ((0, 2), {"T", "R", "O", "C"}),
        ((1, 2), {"V", "C"}),
    ],
)
def test_cross_block_interactions(block_indices, expected_nonzero):
    char_fn = block_interaction_char_fn(PLAYERS, GROUPS, block_indices)
    phi, perm = _owen(char_fn)
    for impl in (phi, perm):
        for p in PLAYERS:
            if p in expected_nonzero:
                assert impl[p] > ATOL
            else:
                assert abs(impl[p]) < ATOL
        assert abs(sum(impl.values()) - 1.0) < ATOL


def test_random_game_subset_equals_permutation_100():
    rng = np.random.default_rng(20260828)
    max_disc = 0.0
    for _ in range(100):
        char_fn = random_char_fn(PLAYERS, rng)
        disc = compare_owen_implementations(PLAYERS, GROUPS, char_fn)
        max_disc = max(max_disc, max(disc.values()))
        assert max(disc.values()) < ATOL
    assert max_disc < ATOL


def test_quotient_shapley_property_100():
    rng = np.random.default_rng(7)
    max_res = 0.0
    for _ in range(100):
        char_fn = random_char_fn(PLAYERS, rng)
        phi = owen_value_subset(PLAYERS, GROUPS, char_fn)
        res = verify_quotient_consistency(PLAYERS, GROUPS, char_fn, phi)
        max_res = max(max_res, *res.values())
        assert res["score_block"] < ATOL
        assert res["value_player"] < ATOL
        assert res["context_player"] < ATOL
    assert max_res < ATOL


def test_global_efficiency_100():
    rng = np.random.default_rng(99)
    max_res = 0.0
    for _ in range(100):
        char_fn = random_char_fn(PLAYERS, rng)
        phi = owen_value_subset(PLAYERS, GROUPS, char_fn)
        res = verify_efficiency(PLAYERS, char_fn, phi)
        max_res = max(max_res, res)
        assert res < ATOL
    assert max_res < ATOL


def test_block_order_invariance_100():
    rng = np.random.default_rng(11)
    max_disc = 0.0
    for _ in range(100):
        char_fn = random_char_fn(PLAYERS, rng)
        disc = owen_block_order_invariance_max(PLAYERS, GROUPS, char_fn)
        max_disc = max(max_disc, disc)
        assert disc < ATOL
    assert max_disc < ATOL


def test_within_block_order_invariance_100():
    rng = np.random.default_rng(13)
    max_disc = 0.0
    for _ in range(100):
        char_fn = random_char_fn(PLAYERS, rng)
        disc = owen_within_block_order_invariance_max(PLAYERS, GROUPS, char_fn)
        max_disc = max(max_disc, disc)
        assert disc < ATOL
    assert max_disc < ATOL


def test_original_fali_special_case():
    t_a, t_b = 1.2, 2.4
    b_a = torch.tensor([0.5, 0.25, -0.1], dtype=torch.float64)
    b_b = torch.tensor([0.9, 0.15, -0.3], dtype=torch.float64)
    res = fali_shapley(t_a, b_a, t_b, b_b)

    char_fn = two_player_attention_char_fn(res.p00, res.p10, res.p01, res.p11)
    generic = shapley_attention_two_player(res.p00, res.p10, res.p01, res.p11)
    assert abs(generic["T"] - res.phi_target) < ATOL
    assert abs(generic["R"] - res.phi_competing) < ATOL

    owen, _ = _owen(char_fn)
    assert abs(owen["T"] - res.phi_target) < ATOL
    assert abs(owen["R"] - res.phi_competing) < ATOL
    assert abs(owen["O"]) < ATOL
    assert abs(owen["V"]) < ATOL
    assert abs(owen["C"]) < ATOL
    assert abs(res.residual) < ATOL


def test_n_cells_constant():
    assert N_EXTENDED_FALI_CELLS == 32


def test_quotient_uses_ordinary_shapley():
    rng = np.random.default_rng(1)
    char_fn = random_char_fn(PLAYERS, rng)
    meta, _ = extended_fali_quotient_game(char_fn)
    assert meta == ("SCORE", "VALUE", "CONTEXT")
    q = extended_fali_quotient_shapley(char_fn)
    phi = owen_value_subset(PLAYERS, GROUPS, char_fn)
    score = sum(phi[p] for p in ("T", "R", "O"))
    assert abs(score - q["SCORE"]) < ATOL
    assert abs(phi["V"] - q["VALUE"]) < ATOL
    assert abs(phi["C"] - q["CONTEXT"]) < ATOL
