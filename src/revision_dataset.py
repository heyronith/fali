"""
Revision-side IOI dataset validation and metadata helpers.

Uses the existing IOIDataset unchanged; adds stricter checks and per-example metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import torch

from ioi_dataset import IOIDataset


def template_id_for_index(example_idx: int, template_style: str) -> str:
    if template_style == "mixed":
        return "A" if example_idx % 2 == 0 else "B"
    return template_style


def validate_ioi_dataset(dataset: IOIDataset, model) -> None:
    """Raise AssertionError if any example violates Phase-1 invariants."""
    tokenizer = model.tokenizer
    to_id = tokenizer.encode(" to")
    assert len(to_id) == 1, "Expected GPT-2 single-token encoding for ' to'"
    expected_end_id = to_id[0]
    eos_id = tokenizer.eos_token_id

    for i in range(dataset.batch_size):
        sample = dataset.samples[i]
        tokens = dataset.clean_tokens[i]
        s_id = dataset.s_token_ids[i].item()
        io_id = dataset.io_token_ids[i].item()

        assert sample["s"] != sample["io"], f"Example {i}: S and IO names must differ"
        assert s_id != io_id, f"Example {i}: S and IO token ids must differ"

        # Single-token name checks (dataset already asserts names, re-check ids)
        assert len(tokenizer.encode(" " + sample["s"])) == 1
        assert len(tokenizer.encode(" " + sample["io"])) == 1

        s_indices = (tokens == s_id).nonzero(as_tuple=True)[0]
        io_indices = (tokens == io_id).nonzero(as_tuple=True)[0]
        assert len(s_indices) == 2, f"Example {i}: S must appear exactly twice"
        assert len(io_indices) == 1, f"Example {i}: IO must appear exactly once"

        end = int(dataset.end[i].item())
        assert 0 <= end < tokens.shape[0], f"Example {i}: end out of range"
        assert tokens[end].item() != eos_id, (
            f"Example {i}: end must not point at EOS/pad token"
        )
        assert tokens[end].item() == expected_end_id, (
            f"Example {i}: expected end token ' to' (id={expected_end_id}), "
            f"got id={tokens[end].item()}"
        )
        assert int(dataset.IO[i].item()) == int(io_indices[0].item())
        assert int(dataset.S1[i].item()) == int(s_indices[0].item())
        assert int(dataset.S2[i].item()) == int(s_indices[1].item())


def build_dataset_metadata(dataset: IOIDataset) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i in range(dataset.batch_size):
        sample = dataset.samples[i]
        rows.append(
            {
                "example_idx": i,
                "template_id": template_id_for_index(i, dataset.template_style),
                "S": sample["s"],
                "IO": sample["io"],
                "prompt": sample["clean_prompt"],
                "end_position": int(dataset.end[i].item()),
                "IO_position": int(dataset.IO[i].item()),
                "S1_position": int(dataset.S1[i].item()),
                "S2_position": int(dataset.S2[i].item()),
            }
        )
    return pd.DataFrame(rows)


def make_validated_ioi_dataset(model, num_samples: int, seed: int, template_style: str = "mixed") -> IOIDataset:
    dataset = IOIDataset(
        model,
        num_samples=num_samples,
        template_style=template_style,
        seed=seed,
    )
    validate_ioi_dataset(dataset, model)
    return dataset
