"""
Canonical IOI head taxonomy for the FALI revision pipeline.

These definitions are fixed for Phase 1. Do not silently change them.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

Head = Tuple[int, int]

# Canonical primary Name Movers (corrected full set)
PRIMARY_NAME_MOVERS: List[Head] = [
    (9, 6),   # L9H6
    (9, 9),   # L9H9
    (10, 0),  # L10H0
]

# Historical partial primary knockout used in the old project
HISTORICAL_PRIMARY_NAME_MOVERS: List[Head] = [
    (9, 9),   # L9H9
    (10, 0),  # L10H0
]

# Canonical Backup Name Movers
BACKUP_NAME_MOVERS: List[Head] = [
    (9, 0),   # L9H0
    (9, 7),   # L9H7
    (10, 1),  # L10H1
    (10, 2),  # L10H2
    (10, 6),  # L10H6
    (10, 10), # L10H10
    (11, 2),  # L11H2
    (11, 9),  # L11H9
]

# Canonical Negative Name Movers
NEGATIVE_NAME_MOVERS: List[Head] = [
    (10, 7),  # L10H7
    (11, 10), # L11H10
]

# Exploratory controls that appeared in the old project
EXTRA_CONTROLS: List[Head] = [
    (10, 3),  # L10H3
    (10, 5),  # L10H5
]


def head_name(layer: int, head: int) -> str:
    return f"L{layer}H{head}"


def head_name_from_pair(pair: Head) -> str:
    return head_name(pair[0], pair[1])


def role_for_head(pair: Head) -> str:
    if pair in PRIMARY_NAME_MOVERS:
        return "primary"
    if pair in BACKUP_NAME_MOVERS:
        return "backup"
    if pair in NEGATIVE_NAME_MOVERS:
        return "negative"
    if pair in EXTRA_CONTROLS:
        return "extra_control"
    return "unknown"


def all_phase1_candidates() -> List[Head]:
    """Ordered candidate list for Phase-1 conditional-gain evaluation."""
    return list(BACKUP_NAME_MOVERS) + list(NEGATIVE_NAME_MOVERS) + list(EXTRA_CONTROLS)


def assert_head_bounds(
    heads: Sequence[Head],
    n_layers: int = 12,
    n_heads: int = 12,
) -> None:
    for layer, head in heads:
        assert 0 <= layer < n_layers, f"Layer {layer} out of bounds [0, {n_layers})"
        assert 0 <= head < n_heads, f"Head {head} out of bounds [0, {n_heads})"
