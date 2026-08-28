"""Synthetic algebra tests for Extended FALI game mathematics."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from src.extended_fali_math import (
    EXTENDED_FALI_GROUPS,
    EXTENDED_FALI_PLAYERS,
    N_EXTENDED_FALI_CELLS,
    additive_char_fn,
    compare_owen_implementations,
    constant_char_fn,
    extended_fali_quotient_game,
    extended_fali_quotient_shapley,
    interaction_char_fn,
    make_char_fn_from_table,
    owen_value_permutation,
    owen_value_subset,
    powerset,
    random_char_fn,
    shapley_attention_two_player,
    shapley_value,
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


def test_tr_pure_interaction_within_score():
    char_fn = interaction_char_fn(PLAYERS, frozenset({"T", "R"}))
    phi, perm = _owen(char_fn)
    assert abs(phi["T"] - 0.5) < ATOL
    assert abs(phi["R"] - 0.5) < ATOL
    assert abs(phi["O"]) < ATOL
    assert abs(phi["V"]) < ATOL
    assert abs(phi["C"]) < ATOL
    assert abs(perm["T"] - 0.5) < ATOL
    assert abs(perm["R"] - 0.5) < ATOL


def test_tc_cross_coalition_interaction():
    """T×C interaction: C joins after entire SCORE block, so C receives full interaction."""
    char_fn = interaction_char_fn(PLAYERS, frozenset({"T", "C"}))
    phi, perm = _owen(char_fn)
    assert abs(phi["T"]) < ATOL
    assert abs(phi["R"]) < ATOL
    assert abs(phi["O"]) < ATOL
    assert abs(phi["V"]) < ATOL
    assert abs(phi["C"] - 1.0) < ATOL
    assert abs(perm["C"] - 1.0) < ATOL


def _brute_hierarchical_owen(char_fn):
    """Independent oracle: same permutation logic, recomputed from scratch."""
    perms = list(itertools.permutations(PLAYERS))  # wrong - must filter
    valid = []
    score, value, context = GROUPS
    for perm in itertools.product(
        itertools.permutations(score),
        itertools.permutations(value),
        itertools.permutations(context),
    ):
        valid.append(tuple(x for grp in perm for x in grp))
    phi = {p: 0.0 for p in PLAYERS}
    for perm in valid:
        seen = []
        for name in perm:
            s = frozenset(seen)
            phi[name] += char_fn(s | {name}) - char_fn(s)
            seen.append(name)
    inv = 1.0 / len(valid)
    return {p: v * inv for p, v in phi.items()}


def test_trc_interaction_oracle():
    char_fn = interaction_char_fn(PLAYERS, frozenset({"T", "R", "C"}))
    phi = owen_value_subset(PLAYERS, GROUPS, char_fn)
    oracle = _brute_hierarchical_owen(char_fn)
    for p in PLAYERS:
        assert abs(phi[p] - oracle[p]) < ATOL


def test_random_game_subset_equals_permutation():
    rng = np.random.default_rng(20260828)
    max_disc = 0.0
    for _ in range(25):
        char_fn = random_char_fn(PLAYERS, rng)
        disc = compare_owen_implementations(PLAYERS, GROUPS, char_fn)
        max_disc = max(max_disc, max(disc.values()))
        assert max(disc.values()) < ATOL
    assert max_disc < ATOL


def test_quotient_property():
    rng = np.random.default_rng(7)
    max_res = 0.0
    for _ in range(20):
        char_fn = random_char_fn(PLAYERS, rng)
        phi = owen_value_subset(PLAYERS, GROUPS, char_fn)
        res = verify_quotient_consistency(PLAYERS, GROUPS, char_fn, phi)
        max_res = max(max_res, *res.values())
        assert res["score_block"] < ATOL
        assert res["value_player"] < ATOL
        assert res["context_player"] < ATOL
    assert max_res < ATOL


def test_global_efficiency():
    rng = np.random.default_rng(99)
    max_res = 0.0
    for _ in range(30):
        char_fn = random_char_fn(PLAYERS, rng)
        phi = owen_value_subset(PLAYERS, GROUPS, char_fn)
        res = verify_efficiency(PLAYERS, char_fn, phi)
        max_res = max(max_res, res)
        assert res < ATOL
    assert max_res < ATOL


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


def test_quotient_meta_names():
    rng = np.random.default_rng(1)
    char_fn = random_char_fn(PLAYERS, rng)
    meta, _ = extended_fali_quotient_game(char_fn)
    assert meta == ("SCORE", "VALUE", "CONTEXT")
    q = extended_fali_quotient_shapley(char_fn)
    phi = owen_value_subset(PLAYERS, GROUPS, char_fn)
    score = sum(phi[p] for p in ("T", "R", "O"))
    assert abs(score - q["SCORE"]) < ATOL
