#!/usr/bin/env python3
"""Compute zero-scheme damage_drop via forward passes only (no 32-cell game)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from extended_fali_ef5_runner import compute_zero_damage_drop_seed, resolve_device  # noqa: E402

EF5_SEEDS = [42, 1, 8, 22]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()
    if args.seed not in EF5_SEEDS:
        raise SystemExit(f"Allowed seeds: {EF5_SEEDS}")

    device = resolve_device(args.device)
    if args.require_cuda and not str(device).startswith("cuda"):
        raise RuntimeError(f"--require-cuda but device={device}")

    out = _REPO / f"results/extended_fali/ef5_ablation_robustness/zero_damage/seed_{args.seed}/damage_drop.csv"
    compute_zero_damage_drop_seed(args.seed, device, _REPO, out)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
