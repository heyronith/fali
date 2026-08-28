"""
Extended FALI primary-circuit damage schemes: zero (historical), matched-mean, matched-resample.

Does not alter historical zero-ablation semantics in revision_interventions.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch

from revision_config import Head, PRIMARY_NAME_MOVERS, assert_head_bounds
from revision_interventions import HookSpec, group_heads_by_layer

DamageScheme = str  # "zero" | "matched_mean" | "matched_resample"

DAMAGE_SCHEMES: Tuple[str, ...] = ("zero", "matched_mean", "matched_resample")
PRIMARY_HEADS: Tuple[Head, ...] = tuple(PRIMARY_NAME_MOVERS)


def _stable_pair_hash(seed: int, recipient: int, donor: int) -> str:
    payload = f"{seed}|{recipient}|{donor}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_mean_ablation_groups(
    meta: pd.DataFrame,
    *,
    min_group_size: int = 2,
) -> pd.DataFrame:
    """
    Leave-one-out mean ablation grouping metadata per example.

    Group key: (template_stratum, end_position). Recipient excluded from its own mean.
    """
    required = {"example_idx", "template_stratum", "end_position"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {missing}")

    rows = []
    for i, row in meta.iterrows():
        ex_i = int(row["example_idx"])
        stratum = row["template_stratum"]
        end_pos = int(row["end_position"])
        peers = meta[
            (meta["template_stratum"] == stratum)
            & (meta["end_position"] == end_pos)
            & (meta["example_idx"] != ex_i)
        ]
        group_size = len(peers)
        if group_size < min_group_size:
            raise ValueError(
                f"example {ex_i}: mean-ablation group size {group_size} < {min_group_size} "
                f"for stratum={stratum!r} end={end_pos}"
            )
        rows.append(
            {
                "example_idx": ex_i,
                "template_stratum": stratum,
                "end_position": end_pos,
                "group_size": group_size,
                "self_excluded": True,
            }
        )
    out = pd.DataFrame(rows).sort_values("example_idx").reset_index(drop=True)
    return out


def compute_matched_mean_replacements(
    clean_z: Dict[Head, torch.Tensor],
    meta: pd.DataFrame,
) -> Dict[Head, torch.Tensor]:
    """
    Per-head leave-one-out mean z tensors [batch, seq, d_head].

    clean_z: mapping (layer, head) -> cached clean hook_z for full batch.
    """
    build_mean_ablation_groups(meta)  # validates group sizes
    batch = len(meta)
    device = next(iter(clean_z.values())).device
    dtype = next(iter(clean_z.values())).dtype
    replacements: Dict[Head, torch.Tensor] = {}

    for head in PRIMARY_HEADS:
        z_all = clean_z[head]
        if z_all.shape[0] != batch:
            raise ValueError("clean_z batch size mismatch")
        out = torch.zeros_like(z_all)
        for i, row in meta.iterrows():
            ex_i = int(row["example_idx"])
            stratum = row["template_stratum"]
            end_pos = int(row["end_position"])
            mask = (
                (meta["template_stratum"] == stratum)
                & (meta["end_position"] == end_pos)
                & (meta["example_idx"] != ex_i)
            )
            peer_idx = meta.loc[mask, "example_idx"].astype(int).tolist()
            mean_z = z_all[peer_idx].mean(dim=0)
            out[ex_i] = mean_z.to(dtype=dtype)
        replacements[head] = out.clone()
    return replacements


def build_resample_donor_map(meta: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Deterministic donor assignment per recipient (metadata-only).

    Raises if any recipient lacks an eligible donor.
    """
    required = {
        "example_idx",
        "template_stratum",
        "end_position",
        "S",
        "IO",
    }
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing columns: {missing}")

    rows = []

    for _, rec in meta.sort_values("example_idx").iterrows():
        ex_i = int(rec["example_idx"])
        candidates: List[int] = []
        for _, don in meta.iterrows():
            ex_j = int(don["example_idx"])
            if ex_j == ex_i:
                continue
            if rec["template_stratum"] != don["template_stratum"]:
                continue
            if int(rec["end_position"]) != int(don["end_position"]):
                continue
            if rec["S"] == don["S"]:
                continue
            if rec["IO"] == don["IO"]:
                continue
            candidates.append(ex_j)

        if not candidates:
            raise ValueError(f"No eligible donor for recipient example_idx={ex_i}")

        candidates.sort(key=lambda j: _stable_pair_hash(seed, ex_i, j))
        donor_idx = candidates[0]
        rows.append(
            {
                "recipient_example_idx": ex_i,
                "donor_example_idx": donor_idx,
                "template_stratum": rec["template_stratum"],
                "end_position": int(rec["end_position"]),
                "recipient_S": rec["S"],
                "recipient_IO": rec["IO"],
                "donor_S": meta.loc[meta["example_idx"] == donor_idx, "S"].iloc[0],
                "donor_IO": meta.loc[meta["example_idx"] == donor_idx, "IO"].iloc[0],
            }
        )

    df = pd.DataFrame(rows)
    if (df["recipient_example_idx"] == df["donor_example_idx"]).any():
        raise ValueError("self-donor in final map")
    audit = {
        "n_self_donors": 0,
        "n_s_collisions": 0,
        "n_io_collisions": 0,
        "n_unmatched": 0,
        "deterministic_map_hash": donor_map_deterministic_hash(df),
    }
    return df, audit


def donor_map_deterministic_hash(donor_df: pd.DataFrame) -> str:
    """SHA256 of canonical donor-map CSV bytes."""
    csv_bytes = donor_df.sort_values("recipient_example_idx").to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def compute_matched_resample_replacements(
    clean_z: Dict[Head, torch.Tensor],
    donor_map: pd.DataFrame,
) -> Dict[Head, torch.Tensor]:
    """Inject donor clean z for all primary heads (same donor per recipient)."""
    batch = len(donor_map)
    replacements: Dict[Head, torch.Tensor] = {}
    for head in PRIMARY_HEADS:
        z_all = clean_z[head]
        out = torch.zeros_like(z_all)
        for _, row in donor_map.iterrows():
            rec = int(row["recipient_example_idx"])
            don = int(row["donor_example_idx"])
            if rec == don:
                raise ValueError(f"self-donor at example {rec}")
            out[rec] = z_all[don].clone()
        replacements[head] = out
    if batch != len(donor_map):
        pass
    return replacements


def _make_composite_replacement_hook(
    layer: int,
    head_indices: Sequence[int],
    replacement_z: Dict[int, torch.Tensor],
) -> callable:
    """
    Replace selected head slices with precomputed z [batch, seq, d_head].

    replacement_z maps head index -> tensor; must not alias live cache tensors.
    """
    head_indices = sorted(set(int(h) for h in head_indices))

    def hook_fn(z: torch.Tensor, hook) -> torch.Tensor:
        assert z.ndim == 4, f"Expected hook_z rank-4, got {tuple(z.shape)}"
        n_heads = z.shape[2]
        for h in head_indices:
            assert 0 <= h < n_heads
            if h not in replacement_z:
                raise ValueError(f"Missing replacement for layer {layer} head {h}")
        z_out = z.clone()
        for h in head_indices:
            rep = replacement_z[h]
            if rep.shape != z_out[:, :, h, :].shape:
                raise ValueError(
                    f"replacement shape {tuple(rep.shape)} != z slice {tuple(z_out[:, :, h, :].shape)}"
                )
            z_out[:, :, h, :] = rep
        return z_out

    return hook_fn


def build_composite_replacement_hooks(
    replacements_by_head: Dict[Head, torch.Tensor],
    *,
    n_layers: int = 12,
    n_heads: int = 12,
) -> List[HookSpec]:
    """
    Build clone-safe composite replacement hooks for primary heads.

    replacements_by_head: (layer, head) -> [batch, seq, d_head] (must be detached clones).
    """
    heads = list(replacements_by_head.keys())
    assert_head_bounds(heads, n_layers=n_layers, n_heads=n_heads)
    grouped = group_heads_by_layer(heads)
    hooks: List[HookSpec] = []
    for layer, head_indices in grouped.items():
        rep_layer = {
            h: replacements_by_head[(layer, h)].detach().clone()
            for h in head_indices
        }
        hook_name = f"blocks.{layer}.attn.hook_z"
        hooks.append((hook_name, _make_composite_replacement_hook(layer, head_indices, rep_layer)))
    names = [n for n, _ in hooks]
    assert len(names) == len(set(names))
    return hooks


def build_damage_hooks(
    scheme: DamageScheme,
    *,
    clean_z: Dict[Head, torch.Tensor] | None = None,
    meta: pd.DataFrame | None = None,
    seed: int | None = None,
    donor_map: pd.DataFrame | None = None,
) -> Tuple[List[HookSpec], pd.DataFrame | None, pd.DataFrame | None, Dict | None]:
    """
    Return forward hooks for primary damage under `scheme`.

    For matched_mean / matched_resample also returns audit artifacts.
    """
    from revision_interventions import build_zero_ablation_hooks

    if scheme == "zero":
        return build_zero_ablation_hooks(PRIMARY_NAME_MOVERS), None, None, None

    if clean_z is None or meta is None:
        raise ValueError(f"{scheme} requires clean_z and meta")

    if scheme == "matched_mean":
        groups = build_mean_ablation_groups(meta)
        reps = compute_matched_mean_replacements(clean_z, meta)
        hooks = build_composite_replacement_hooks(reps)
        return hooks, groups, None, None

    if scheme == "matched_resample":
        if seed is None:
            raise ValueError("matched_resample requires seed")
        donor_df, audit = build_resample_donor_map(meta, seed)
        if donor_map is not None:
            if not donor_df.equals(donor_map.sort_values("recipient_example_idx").reset_index(drop=True)):
                raise ValueError("donor_map mismatch vs deterministic regeneration")
        reps = compute_matched_resample_replacements(clean_z, donor_df)
        hooks = build_composite_replacement_hooks(reps)
        audit = dict(audit)
        audit["deterministic_map_hash"] = donor_map_deterministic_hash(donor_df)
        return hooks, None, donor_df, audit

    raise ValueError(f"Unknown damage scheme: {scheme}")


def extract_primary_clean_z(cache, batch_size: int) -> Dict[Head, torch.Tensor]:
    """Extract clean hook_z for all primary heads from a run_with_cache result."""
    out: Dict[Head, torch.Tensor] = {}
    for layer, head in PRIMARY_HEADS:
        z = cache[f"blocks.{layer}.attn.hook_z"][:, :, head, :].contiguous()
        if z.shape[0] != batch_size:
            raise ValueError("batch size mismatch in primary z cache")
        out[(layer, head)] = z.clone()
    return out


def verify_hook_preserves_nonprimary_slices(
    z_in: torch.Tensor,
    hooks: Sequence[HookSpec],
    layer: int,
    nonprimary_heads: Sequence[int],
) -> float:
    """
    Max change on non-primary head slices after applying layer hook_z once.

    z_in: [batch, seq, n_heads, d_head] clean tensor (cloned before call).
    """
    hook_name = f"blocks.{layer}.attn.hook_z"
    hook_fn = None
    for name, fn in hooks:
        if name == hook_name:
            hook_fn = fn
            break
    if hook_fn is None:
        return 0.0
    z_out = hook_fn(z_in.clone(), None)
    max_diff = 0.0
    for h in nonprimary_heads:
        diff = (z_out[:, :, h, :] - z_in[:, :, h, :]).abs().max().item()
        max_diff = max(max_diff, float(diff))
    return max_diff
