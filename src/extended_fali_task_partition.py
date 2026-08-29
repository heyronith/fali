"""
Generic Extended-FALI score partition API for task-specific T/R/O masks.
"""

from __future__ import annotations

from typing import Sequence

import torch

from extended_fali_interventions import ScorePartition, build_batched_score_partition, validate_score_partition


def build_task_score_partition(
    seq_len: int,
    final_queries: torch.Tensor,
    target_keys: torch.Tensor,
    *,
    valid_lengths: torch.Tensor | None = None,
    device: torch.device | str | None = None,
) -> ScorePartition:
    """
    Build structural T/R/O partition from per-example final query, target key,
    and optional valid lengths (default final_query + 1).
    """
    partition = build_batched_score_partition(
        seq_len,
        final_queries,
        target_keys,
        valid_lengths=valid_lengths,
        device=device,
    )
    validate_score_partition(partition)
    return partition


def build_induction_partition(
    final_queries: torch.Tensor,
    target_keys: torch.Tensor,
    *,
    valid_lengths: torch.Tensor,
    seq_len: int,
    device: torch.device | str | None = None,
) -> ScorePartition:
    """Induction task partition: T=(q,k), valid_length=q+1."""
    return build_task_score_partition(
        seq_len,
        final_queries,
        target_keys,
        valid_lengths=valid_lengths,
        device=device,
    )
