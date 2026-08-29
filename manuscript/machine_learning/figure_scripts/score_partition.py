"""FALI structural score-matrix partition for schematic figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

Player = Literal["T", "R", "O", "MASK"]

ILLUSTRATIVE_Q_STAR = 5
ILLUSTRATIVE_K_STAR = 2
ILLUSTRATIVE_N = 6


@dataclass(frozen=True)
class PartitionSummary:
    n: int
    q_star: int
    k_star: int
    n_T: int
    n_R: int
    n_O: int
    n_mask: int
    n_valid: int
    final_row: List[Player]


def classify_score_cell(q: int, k: int, *, q_star: int, k_star: int) -> Player:
    """Partition rule: valid(q,k) iff k <= q."""
    if k > q:
        return "MASK"
    if q == q_star and k == k_star:
        return "T"
    if q == q_star and k <= q_star:
        return "R"
    if q < q_star:
        return "O"
    raise ValueError(f"Unclassified valid cell ({q}, {k}) for q*={q_star}, k*={k_star}")


def build_score_partition(
    n: int = ILLUSTRATIVE_N,
    *,
    q_star: int = ILLUSTRATIVE_Q_STAR,
    k_star: int = ILLUSTRATIVE_K_STAR,
) -> Dict[Tuple[int, int], Player]:
    if q_star >= n or k_star > q_star:
        raise ValueError("Invalid illustrative dimensions")
    grid: Dict[Tuple[int, int], Player] = {}
    for q in range(n):
        for k in range(n):
            grid[(q, k)] = classify_score_cell(q, k, q_star=q_star, k_star=k_star)
    return grid


def summarize_partition(
    grid: Dict[Tuple[int, int], Player],
    *,
    n: int = ILLUSTRATIVE_N,
    q_star: int = ILLUSTRATIVE_Q_STAR,
    k_star: int = ILLUSTRATIVE_K_STAR,
) -> PartitionSummary:
    counts = {"T": 0, "R": 0, "O": 0, "MASK": 0}
    valid: List[Tuple[int, int, Player]] = []
    for (q, k), player in grid.items():
        counts[player] += 1
        if player != "MASK":
            valid.append((q, k, player))
    final_row = [grid[(q_star, k)] for k in range(n)]
    players = {p for _, _, p in valid}
    if len(players & {"T", "R", "O"}) != 3:
        raise ValueError("Valid cells must include T, R, and O")
    if len({(q, k) for q, k, _ in valid}) != len(valid):
        raise ValueError("Duplicate valid coordinates")
    union_valid = counts["T"] + counts["R"] + counts["O"]
    lower_tri = n * (n + 1) // 2
    if union_valid != lower_tri:
        raise ValueError(f"T+R+O ({union_valid}) != valid lower triangle ({lower_tri})")
    if counts["MASK"] != n * n - lower_tri:
        raise ValueError("Mask count mismatch")
    return PartitionSummary(
        n=n,
        q_star=q_star,
        k_star=k_star,
        n_T=counts["T"],
        n_R=counts["R"],
        n_O=counts["O"],
        n_mask=counts["MASK"],
        n_valid=union_valid,
        final_row=final_row,
    )


def validate_illustrative_partition() -> PartitionSummary:
    grid = build_score_partition()
    summary = summarize_partition(grid)
    if summary.n_T != 1:
        raise AssertionError("Expected exactly one T cell")
    if summary.n_R != summary.q_star:
        raise AssertionError(f"Expected {summary.q_star} R cells, got {summary.n_R}")
    if grid[(summary.q_star, summary.q_star)] != "R":
        raise AssertionError("(q*, q*) must be R, not mask")
    for q in range(summary.q_star):
        for k in range(q + 1):
            if grid[(q, k)] != "O":
                raise AssertionError(f"Expected O at ({q}, {k})")
    for q in range(summary.n):
        for k in range(summary.n):
            if k > q and grid[(q, k)] != "MASK":
                raise AssertionError(f"Expected MASK at ({q}, {k})")
    return summary
