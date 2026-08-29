#!/usr/bin/env python3
"""EF-6 discovery: G0 behavior + head-pair selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from extended_fali_ef6_runner import EF6RunConfig, PREREG_SHA, resolve_device, run_ef6_discovery  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO / "results/extended_fali/ef6_pythia_induction/discovery",
    )
    args = p.parse_args()
    device = resolve_device(args.device)
    if args.require_cuda and device != "cuda":
        raise SystemExit("CUDA required")
    report = run_ef6_discovery(
        EF6RunConfig(
            phase="discovery",
            seed=42,
            out_dir=args.output_dir,
            device=device,
            preregistration_sha=PREREG_SHA,
            repo_root=_REPO,
        )
    )
    if report.get("stop_reason"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
