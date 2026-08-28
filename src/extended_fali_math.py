"""
Extended / Causal FALI — pure game-theoretic mathematics.

No TransformerLens dependency. Implements exact Shapley and hierarchical Owen
values for the frozen five-player coalition structure:

    Pi = { SCORE = {T, R, O}, VALUE = {V}, CONTEXT = {C} }

Players are binary intervention factors (0 = A state, 1 = B state). A
characteristic function v(S) gives the utility U when exactly players in S
are at state 1 and all others at state 0.
"""

from __future__ import annotations

import itertools
import math
from typing import Callable, Dict, FrozenSet, Iterable, List, Mapping, Sequence, Tuple, Union

import numpy as np

Player = str
Coalition = FrozenSet[Player]
CharFn = Callable[[Coalition], float]

# Frozen Extended FALI player set and coalition hierarchy (prospective, not data-driven).
EXTENDED_FALI_PLAYERS: Tuple[Player, ...] = ("T", "R", "O", "V", "C")
EXTENDED_FALI_GROUPS: Tuple[Tuple[Player, ...], ...] = (
    ("T", "R", "O"),
    ("V",),
    ("C",),
)
EXTENDED_FALI_META: Tuple[Player, ...] = ("SCORE", "VALUE", "CONTEXT")
N_EXTENDED_FALI_CELLS: int = 2 ** len(EXTENDED_FALI_PLAYERS)  # 32

_PLAYER_TO_META: Dict[Player, Player] = {
    "T": "SCORE",
    "R": "SCORE",
    "O": "SCORE",
    "V": "VALUE",
    "C": "CONTEXT",
}


def powerset(items: Sequence[Player]) -> List[Coalition]:
    """All coalitions over items in deterministic lexicographic subset order."""
    ordered = tuple(items)
    out: List[Coalition] = []
    for r in range(len(ordered) + 1):
        for combo in itertools.combinations(ordered, r):
            out.append(frozenset(combo))
    return out


def _validate_players(players: Sequence[Player]) -> Tuple[Player, ...]:
    if len(players) != len(set(players)):
        raise ValueError(f"players must be unique: {players}")
    return tuple(players)


def _validate_groups(
    players: Sequence[Player],
    groups: Sequence[Sequence[Player]],
) -> Tuple[Tuple[Player, ...], ...]:
    flat: List[Player] = []
    seen = set()
    norm_groups: List[Tuple[Player, ...]] = []
    for group in groups:
        g = tuple(group)
        if not g:
            raise ValueError("groups must be non-empty")
        if len(g) != len(set(g)):
            raise ValueError(f"duplicate player within group: {g}")
        for p in g:
            if p in seen:
                raise ValueError(f"player {p} appears in multiple groups")
            seen.add(p)
        flat.extend(g)
        norm_groups.append(g)
    if set(flat) != set(players):
        raise ValueError(
            f"groups must partition players exactly; got groups={flat}, players={players}"
        )
    return tuple(norm_groups)


def _factorial(n: int) -> int:
    if n < 0:
        raise ValueError(f"factorial domain error: {n}")
    return math.factorial(n)


def coalition_from_mask(
    players: Sequence[Player],
    mask: int,
) -> Coalition:
    """Bit i of mask corresponds to players[i] (LSB = players[0])."""
    p = _validate_players(players)
    if mask < 0 or mask >= (1 << len(p)):
        raise ValueError(f"mask {mask} out of range for {len(p)} players")
    return frozenset(p[i] for i in range(len(p)) if (mask >> i) & 1)


def mask_from_coalition(players: Sequence[Player], coalition: Coalition) -> int:
    p = _validate_players(players)
    unknown = coalition - set(p)
    if unknown:
        raise ValueError(f"coalition contains unknown players: {unknown}")
    mask = 0
    for i, name in enumerate(p):
        if name in coalition:
            mask |= 1 << i
    return mask


def make_char_fn_from_table(
    players: Sequence[Player],
    values: Mapping[Union[Coalition, Tuple[int, ...], int], float],
) -> CharFn:
    """
    Build v(S) from a lookup table.

    Keys may be coalitions, 0/1 tuples aligned with players, or integer masks.
    """
    p = _validate_players(players)
    table: Dict[Coalition, float] = {}
    for key, val in values.items():
        if isinstance(key, frozenset):
            coalition = key
        elif isinstance(key, int):
            coalition = coalition_from_mask(p, key)
        elif isinstance(key, tuple):
            if len(key) == 0:
                coalition = frozenset()
            elif len(key) != len(p):
                raise ValueError(f"tuple key length {len(key)} != {len(p)} players")
            else:
                coalition = frozenset(p[i] for i, bit in enumerate(key) if bit)
        else:
            raise TypeError(f"unsupported table key type: {type(key)}")
        table[coalition] = float(val)

    def v(coalition: Coalition) -> float:
        extra = coalition - set(p)
        if extra:
            raise ValueError(f"unknown coalition members: {extra}")
        key = frozenset(coalition)
        if key not in table:
            raise KeyError(f"coalition {sorted(key)} missing from value table")
        return table[key]

    return v


def shapley_value(
    players: Sequence[Player],
    char_fn: CharFn,
) -> Dict[Player, float]:
    """Exact Shapley value for cooperative game (v(∅) need not be zero)."""
    p = _validate_players(players)
    n = len(p)
    phi = {name: 0.0 for name in p}
    for i, name in enumerate(p):
        others = [p[j] for j in range(n) if j != i]
        for r in range(len(others) + 1):
            for subset in itertools.combinations(others, r):
                s = frozenset(subset)
                weight = _factorial(len(s)) * _factorial(n - len(s) - 1) / _factorial(n)
                phi[name] += weight * (char_fn(s | {name}) - char_fn(s))
    return phi


def _hierarchical_permutations(
    groups: Sequence[Sequence[Player]],
) -> Iterable[Tuple[Player, ...]]:
    group_perms = [itertools.permutations(g) for g in groups]
    for combo in itertools.product(*group_perms):
        yield tuple(player for grp in combo for player in grp)


def owen_value_permutation(
    players: Sequence[Player],
    groups: Sequence[Sequence[Player]],
    char_fn: CharFn,
) -> Dict[Player, float]:
    """
    Hierarchical Owen value via brute-force enumeration of valid permutations.

    Permutations respect group order (earlier groups fully precede later groups)
    and allow arbitrary order within each group.
    """
    p = _validate_players(players)
    norm_groups = _validate_groups(p, groups)
    perms = list(_hierarchical_permutations(norm_groups))
    if not perms:
        raise ValueError("empty groups produce no permutations")
    phi = {name: 0.0 for name in p}
    for perm in perms:
        seen: List[Player] = []
        for name in perm:
            s = frozenset(seen)
            phi[name] += char_fn(s | {name}) - char_fn(s)
            seen.append(name)
    inv = 1.0 / float(len(perms))
    return {name: val * inv for name, val in phi.items()}


def owen_value_subset(
    players: Sequence[Player],
    groups: Sequence[Sequence[Player]],
    char_fn: CharFn,
) -> Dict[Player, float]:
    """
    Hierarchical Owen value via closed-form subset weights.

    For player i in group G_k, all players from earlier groups are always in
    the predecessor coalition; within G_k, predecessors range over subsets of
    G_k \\ {i} with standard factorial weights.
    """
    p = _validate_players(players)
    norm_groups = _validate_groups(p, groups)
    phi = {name: 0.0 for name in p}
    for k, group in enumerate(norm_groups):
        earlier = frozenset(player for g in norm_groups[:k] for player in g)
        gk = tuple(group)
        gk_size = len(gk)
        for player in gk:
            others = [x for x in gk if x != player]
            m = len(others)
            acc = 0.0
            for r in range(m + 1):
                for subset in itertools.combinations(others, r):
                    tset = frozenset(subset)
                    weight = _factorial(len(tset)) * _factorial(m - len(tset))
                    before = earlier | tset
                    acc += weight * (char_fn(before | {player}) - char_fn(before))
            phi[player] = acc / float(_factorial(gk_size))
    return phi


def owen_value(
    players: Sequence[Player],
    groups: Sequence[Sequence[Player]],
    char_fn: CharFn,
    *,
    method: str = "subset",
) -> Dict[Player, float]:
    if method == "subset":
        return owen_value_subset(players, groups, char_fn)
    if method == "permutation":
        return owen_value_permutation(players, groups, char_fn)
    raise ValueError(f"unknown owen method: {method}")


def quotient_game(
    players: Sequence[Player],
    groups: Sequence[Sequence[Player]],
    char_fn: CharFn,
) -> Tuple[Tuple[Player, ...], CharFn]:
    """
    Collapse players into meta-players (one per group) for the quotient game.

    Returns (meta_player_names, v_quotient) where v_quotient(S) evaluates the
    original game with every member of each selected group at state 1.
    """
    p = _validate_players(players)
    norm_groups = _validate_groups(p, groups)
    meta_names = tuple(f"G{k}" for k in range(len(norm_groups)))
    group_members = [frozenset(g) for g in norm_groups]

    def v_q(coalition: Coalition) -> float:
        unknown = coalition - set(meta_names)
        if unknown:
            raise ValueError(f"unknown meta players: {unknown}")
        full = set()
        for idx, meta in enumerate(meta_names):
            if meta in coalition:
                full.update(group_members[idx])
        return char_fn(frozenset(full))

    return meta_names, v_q


def quotient_shapley_fixed_order(
    players: Sequence[Player],
    groups: Sequence[Sequence[Player]],
    char_fn: CharFn,
) -> Dict[Player, float]:
    """
    Meta-level attributions under the frozen group order (not full 3! Shapley).

    With hierarchy G1 ≺ G2 ≺ G3, the SCORE meta-player receives
    v(G1) − v(∅), VALUE receives v(G1∪G2) − v(G1), etc.
    """
    _, v_q = quotient_game(players, groups, char_fn)
    norm_groups = _validate_groups(_validate_players(players), groups)
    meta = tuple(f"G{k}" for k in range(len(norm_groups)))
    phi: Dict[Player, float] = {}
    prefix: Coalition = frozenset()
    for name in meta:
        phi[name] = v_q(prefix | {name}) - v_q(prefix)
        prefix = prefix | {name}
    return phi


def extended_fali_quotient_shapley(char_fn: CharFn) -> Dict[Player, float]:
    """Named meta labels for the frozen Extended FALI quotient attributions."""
    internal = quotient_shapley_fixed_order(
        EXTENDED_FALI_PLAYERS,
        EXTENDED_FALI_GROUPS,
        char_fn,
    )
    return {
        "SCORE": internal["G0"],
        "VALUE": internal["G1"],
        "CONTEXT": internal["G2"],
    }


def extended_fali_quotient_game(char_fn: CharFn) -> Tuple[Tuple[Player, ...], CharFn]:
    """Quotient game for the frozen Extended FALI hierarchy with named meta labels."""
    meta_internal, v_q = quotient_game(
        EXTENDED_FALI_PLAYERS,
        EXTENDED_FALI_GROUPS,
        char_fn,
    )
    rename = {
        "G0": "SCORE",
        "G1": "VALUE",
        "G2": "CONTEXT",
    }
    meta_named = tuple(rename[m] for m in meta_internal)

    def v_named(coalition: Coalition) -> float:
        internal = frozenset(
            {"SCORE": "G0", "VALUE": "G1", "CONTEXT": "G2"}[m] for m in coalition
        )
        return v_q(internal)

    return meta_named, v_named


def verify_efficiency(
    players: Sequence[Player],
    char_fn: CharFn,
    attributions: Mapping[Player, float],
    *,
    atol: float = 1e-12,
) -> float:
    """
    Return absolute efficiency residual:

        sum(phi) - (v(N) - v(empty))
    """
    p = _validate_players(players)
    total = sum(float(attributions[name]) for name in p)
    grand = char_fn(frozenset(p))
    empty = char_fn(frozenset())
    return abs(total - (grand - empty))


def verify_quotient_consistency(
    players: Sequence[Player],
    groups: Sequence[Sequence[Player]],
    char_fn: CharFn,
    owen_phi: Mapping[Player, float],
    *,
    atol: float = 1e-12,
) -> Dict[str, float]:
    """
    Return absolute residuals for hierarchical Owen vs quotient Shapley blocks.

    Keys:
      - score_block
      - value_player
      - context_player
    """
    p = _validate_players(players)
    norm_groups = _validate_groups(p, groups)
    meta_players, v_q = quotient_game(p, norm_groups, char_fn)
    q_phi = quotient_shapley_fixed_order(p, norm_groups, char_fn)

    score_players = norm_groups[0]
    score_sum = sum(float(owen_phi[x]) for x in score_players)
    value_player = norm_groups[1][0]
    context_player = norm_groups[2][0]

    return {
        "score_block": abs(score_sum - q_phi[meta_players[0]]),
        "value_player": abs(float(owen_phi[value_player]) - q_phi[meta_players[1]]),
        "context_player": abs(float(owen_phi[context_player]) - q_phi[meta_players[2]]),
    }


def compare_owen_implementations(
    players: Sequence[Player],
    groups: Sequence[Sequence[Player]],
    char_fn: CharFn,
) -> Dict[Player, float]:
    """Return per-player absolute discrepancy between subset and permutation Owen."""
    subset = owen_value_subset(players, groups, char_fn)
    perm = owen_value_permutation(players, groups, char_fn)
    return {p: abs(subset[p] - perm[p]) for p in players}


def constant_char_fn(players: Sequence[Player], value: float) -> CharFn:
    """v(S) = value for every coalition (zero total gain)."""
    _validate_players(players)
    val = float(value)

    def v(coalition: Coalition) -> float:
        return val

    return v


def additive_char_fn(
    players: Sequence[Player],
    contributions: Mapping[Player, float],
) -> CharFn:
    """v(S) = sum_{i in S} a_i (purely additive game)."""
    p = _validate_players(players)
    coeffs = {k: float(v) for k, v in contributions.items()}
    unknown = set(coeffs) - set(p)
    if unknown:
        raise ValueError(f"unknown contribution keys: {unknown}")

    def v(coalition: Coalition) -> float:
        return sum(coeffs[name] for name in coalition if name in coeffs)

    return v


def interaction_char_fn(
    players: Sequence[Player],
    required: Coalition,
    value: float = 1.0,
) -> CharFn:
    """v(S) = value iff required subset is contained in S, else 0."""
    p = _validate_players(players)
    req = frozenset(required)
    extra = req - set(p)
    if extra:
        raise ValueError(f"required players not in player set: {extra}")
    val = float(value)

    def v(coalition: Coalition) -> float:
        return val if req.issubset(coalition) else 0.0

    return v


def random_char_fn(
    players: Sequence[Player],
    rng: np.random.Generator,
) -> CharFn:
    """Uniform random values on all coalitions (for generic agreement tests)."""
    p = _validate_players(players)
    table = {coal: float(rng.uniform(-3.0, 3.0)) for coal in powerset(p)}

    def v(coalition: Coalition) -> float:
        key = frozenset(coalition)
        if key not in table:
            raise ValueError(f"unknown coalition: {key}")
        return table[key]

    return v


def two_player_attention_char_fn(
    p00: float,
    p10: float,
    p01: float,
    p11: float,
) -> CharFn:
    """
    Attention-probability special case on {T,R} with O,V,C inactive.

    Coalition membership selects B-state for T (target score) and R (competing block).
    """
    table = {
        frozenset(): p00,
        frozenset({"T"}): p10,
        frozenset({"R"}): p01,
        frozenset({"T", "R"}): p11,
    }

    def v(coalition: Coalition) -> float:
        key = frozenset(x for x in coalition if x in {"T", "R"})
        if key not in table:
            raise KeyError(f"unsupported coalition for 2-player attention game: {key}")
        return table[key]

    return v


def shapley_attention_two_player(
    p00: float,
    p10: float,
    p01: float,
    p11: float,
) -> Dict[str, float]:
    """Generic Shapley on {T,R} for the attention-probability game."""
    char_fn = two_player_attention_char_fn(p00, p10, p01, p11)
    return shapley_value(("T", "R"), char_fn)
