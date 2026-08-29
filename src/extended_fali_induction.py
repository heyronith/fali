"""
Synthetic repeated-token induction dataset for EF-6 Pythia transfer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from extended_fali_task_partition import build_induction_partition

# Frozen EF-6 dataset constants (preregistered).
TOKEN_POOL_LOW = 1000
TOKEN_POOL_HIGH = 10000
BASE_LEN = 32
QUERY_OFFSETS = (8, 16, 24)
EXAMPLES_PER_OFFSET = 128
DISCOVERY_SEED = 42
CONFIRMATION_SEEDS = (1, 8, 22)
PADDING_TOKEN_ID = 0


@dataclass
class InductionDataset:
    repeated_tokens: torch.Tensor  # [B, Lmax] int64
    control_tokens: torch.Tensor
    query_pos: torch.Tensor  # [B]
    target_key: torch.Tensor  # [B]
    target_token_id: torch.Tensor  # [B]
    valid_length: torch.Tensor  # [B]
    query_offset: torch.Tensor  # [B]
    metadata: pd.DataFrame

    @property
    def batch_size(self) -> int:
        return int(self.repeated_tokens.shape[0])

    @property
    def seq_len(self) -> int:
        return int(self.repeated_tokens.shape[1])

    def partition(self, device: torch.device | str) -> object:
        return build_induction_partition(
            self.query_pos.to(device),
            self.target_key.to(device),
            valid_lengths=self.valid_length.to(device),
            seq_len=self.seq_len,
            device=device,
        )


def _example_lengths(offset: int) -> Tuple[int, int, int]:
    q = BASE_LEN + offset
    k = offset + 1
    seq_len = BASE_LEN + offset + 1
    valid_length = q + 1
    return q, k, seq_len


def generate_induction_metadata(
    *,
    seed: int,
    examples_per_offset: int = EXAMPLES_PER_OFFSET,
    offsets: Sequence[int] = QUERY_OFFSETS,
    smoke_deterministic: bool = False,
    smoke_n: int = 8,
) -> pd.DataFrame:
    """Generate dataset metadata rows; one row per (example, offset)."""
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    example_idx = 0
    n_examples = smoke_n if smoke_deterministic else examples_per_offset
    for offset in offsets:
        for _ in range(n_examples):
            ids = rng.choice(
                np.arange(TOKEN_POOL_LOW, TOKEN_POOL_HIGH, dtype=np.int64),
                size=64,
                replace=False,
            )
            x = ids[:BASE_LEN]
            y = ids[BASE_LEN:]
            j = int(offset)
            q, k, seq_len = _example_lengths(j)
            target = int(x[j + 1])
            repeated = np.concatenate([x, x[: j + 1]])
            control = np.concatenate([y, x[: j + 1]])
            valid_length = q + 1
            assert len(repeated) == seq_len
            assert len(control) == seq_len
            rows.append(
                {
                    "seed": seed,
                    "example_idx": example_idx,
                    "query_offset": j,
                    "x_block_json": json.dumps(x.tolist()),
                    "y_block_json": json.dumps(y.tolist()),
                    "repeated_token_ids_json": json.dumps(repeated.tolist()),
                    "control_token_ids_json": json.dumps(control.tolist()),
                    "target_token_id": target,
                    "query_pos": q,
                    "target_key": k,
                    "seq_len": seq_len,
                    "valid_length": valid_length,
                    "padding_start": valid_length,
                }
            )
            example_idx += 1
    return pd.DataFrame(rows)


def validate_induction_metadata(meta: pd.DataFrame) -> None:
    """Assert all preregistered dataset invariants."""
    for _, row in meta.iterrows():
        x = np.array(json.loads(row["x_block_json"]), dtype=np.int64)
        y = np.array(json.loads(row["y_block_json"]), dtype=np.int64)
        repeated = np.array(json.loads(row["repeated_token_ids_json"]), dtype=np.int64)
        control = np.array(json.loads(row["control_token_ids_json"]), dtype=np.int64)
        j = int(row["query_offset"])
        q = int(row["query_pos"])
        k = int(row["target_key"])
        target = int(row["target_token_id"])
        vl = int(row["valid_length"])
        seq_len = int(row["seq_len"])

        assert len(x) == BASE_LEN
        assert len(y) == BASE_LEN
        all_ids = np.concatenate([x, y])
        assert len(np.unique(all_ids)) == 64
        assert all_ids.min() >= TOKEN_POOL_LOW and all_ids.max() < TOKEN_POOL_HIGH
        assert len(set(x.tolist()) & set(y.tolist())) == 0
        assert q == BASE_LEN + j
        assert k == j + 1
        assert target == int(x[j + 1])
        assert k < q
        assert vl == q + 1
        assert seq_len == BASE_LEN + j + 1
        assert len(repeated) == seq_len
        assert len(control) == seq_len

        suffix = repeated[BASE_LEN : q + 1]
        assert target not in suffix.tolist()
        assert int(x[j]) == int(repeated[q])
        control_prefix = control[:BASE_LEN]
        assert target not in control_prefix.tolist()
        assert int(x[j]) not in set(control_prefix.tolist())

        pad_start = int(row["padding_start"])
        assert pad_start == vl


def metadata_to_tensors(meta: pd.DataFrame, *, max_seq_len: int | None = None) -> InductionDataset:
    validate_induction_metadata(meta)
    if max_seq_len is None:
        max_seq_len = int(meta["seq_len"].max())
    batch = len(meta)
    rep = torch.full((batch, max_seq_len), PADDING_TOKEN_ID, dtype=torch.long)
    ctl = torch.full((batch, max_seq_len), PADDING_TOKEN_ID, dtype=torch.long)
    query_pos = torch.zeros(batch, dtype=torch.long)
    target_key = torch.zeros(batch, dtype=torch.long)
    target_token_id = torch.zeros(batch, dtype=torch.long)
    valid_length = torch.zeros(batch, dtype=torch.long)
    query_offset = torch.zeros(batch, dtype=torch.long)

    for i, row in meta.iterrows():
        idx = int(i) if isinstance(i, int) else meta.index.get_loc(i)
        r = np.array(json.loads(row["repeated_token_ids_json"]), dtype=np.int64)
        c = np.array(json.loads(row["control_token_ids_json"]), dtype=np.int64)
        sl = len(r)
        rep[idx, :sl] = torch.tensor(r, dtype=torch.long)
        ctl[idx, :sl] = torch.tensor(c, dtype=torch.long)
        query_pos[idx] = int(row["query_pos"])
        target_key[idx] = int(row["target_key"])
        target_token_id[idx] = int(row["target_token_id"])
        valid_length[idx] = int(row["valid_length"])
        query_offset[idx] = int(row["query_offset"])
        q = int(row["query_pos"])
        vl = int(row["valid_length"])
        if sl < max_seq_len:
            assert (rep[idx, sl:] == PADDING_TOKEN_ID).all()
            assert sl == vl

    return InductionDataset(
        repeated_tokens=rep,
        control_tokens=ctl,
        query_pos=query_pos,
        target_key=target_key,
        target_token_id=target_token_id,
        valid_length=valid_length,
        query_offset=query_offset,
        metadata=meta.reset_index(drop=True),
    )


def tensor_sha256(dataset: InductionDataset) -> str:
    h = hashlib.sha256()
    h.update(dataset.repeated_tokens.numpy().tobytes())
    h.update(dataset.control_tokens.numpy().tobytes())
    h.update(dataset.query_pos.numpy().tobytes())
    h.update(dataset.target_token_id.numpy().tobytes())
    return h.hexdigest()


def save_dataset_artifacts(dataset: InductionDataset, out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "dataset_metadata.csv"
    dataset.metadata.to_csv(meta_path, index=False)
    sha = tensor_sha256(dataset)
    manifest = {
        "repeated_tokens_sha256": sha,
        "batch_size": dataset.batch_size,
        "max_seq_len": dataset.seq_len,
    }
    (out_dir / "dataset_tensors_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"dataset_metadata": str(meta_path), "tensor_sha256": sha}
