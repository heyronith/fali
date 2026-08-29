#!/usr/bin/env python3
"""EF-6 held-out confirmation seed run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from extended_fali_ef6_runner import EF6RunConfig, PREREG_SHA, resolve_device, run_ef6_confirmation  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True, choices=[1, 8, 22])
    p.add_argument("--device", default="auto")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--selection-lock", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = p.parse_args()
    device = resolve_device(args.device)
    if args.require_cuda and device != "cuda":
        raise SystemExit("CUDA required")
    out = args.output_dir or (
        _REPO / f"results/extended_fali/ef6_pythia_induction/confirmation/seed_{args.seed}"
    )
    val = run_ef6_confirmation(
        EF6RunConfig(
            phase="confirmation",
            seed=args.seed,
            out_dir=out,
            device=device,
            preregistration_sha=PREREG_SHA,
            repo_root=_REPO,
            selection_lock_path=args.selection_lock,
        )
    )
    return 0 if val.get("overall_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
