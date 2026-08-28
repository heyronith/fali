#!/usr/bin/env python3
"""EF-4 — Extended FALI canonical fine-score confirmation (one seed per invocation)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from extended_fali_gpt2_runner import RunConfig, resolve_device, run_extended_fali_seed  # noqa: E402

EF4_SEEDS = {42, 1, 8, 22}
N_EXPECTED = 900
NUM_PER_STRATUM = 30
EF4_HEADS = ("L9H7", "L9H0", "L10H2", "L10H10")
PREREGISTRATION_SHA = "2661459f047f65fabc418993d00e3bb06cc0b3b6"


def assert_safe_output_dir(output_dir: Path, repo: Path) -> None:
    rel = str(output_dir.resolve().relative_to(repo.resolve())).replace("\\", "/")
    if "results/extended_fali/ef4_canonical" not in rel:
        raise ValueError(f"Refusing output path: {rel}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num-samples", type=int, default=N_EXPECTED)
    p.add_argument("--num-per-stratum", type=int, default=NUM_PER_STRATUM)
    p.add_argument("--device", default="auto")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    if args.seed not in EF4_SEEDS:
        raise SystemExit(f"EF-4 allows seeds {sorted(EF4_SEEDS)} only")
    if args.num_samples != N_EXPECTED:
        raise SystemExit(f"EF-4 requires N={N_EXPECTED}")
    if args.num_per_stratum != NUM_PER_STRATUM:
        raise SystemExit(f"EF-4 requires num_per_stratum={NUM_PER_STRATUM}")

    out = Path(args.output_dir) if args.output_dir else _REPO / f"results/extended_fali/ef4_canonical/seed_{args.seed}"
    if not out.is_absolute():
        out = _REPO / out
    assert_safe_output_dir(out, _REPO)

    device = resolve_device(args.device)
    if args.require_cuda and not str(device).startswith("cuda"):
        raise RuntimeError(f"--require-cuda but device={device}")

    ref_meta = _REPO / f"results/revision_phase4/full/seed_{args.seed}/dataset_metadata.csv"
    hist_factorial = _REPO / f"results/revision_phase4/full/seed_{args.seed}/factorial_raw.csv"
    if not ref_meta.exists():
        raise FileNotFoundError(ref_meta)
    if not hist_factorial.exists():
        raise FileNotFoundError(hist_factorial)

    cfg = RunConfig(
        phase="EF-4",
        num_samples=args.num_samples,
        seed=args.seed,
        out_dir=out,
        reference_meta=ref_meta,
        historical_factorial=hist_factorial,
        required_hash_paths=[f"results/revision_phase4/full/seed_{args.seed}/factorial_raw.csv"],
        device=device,
        run_smoke_extras=False,
        compress_cells=True,
        dataset_kind="canonical",
        num_per_stratum=args.num_per_stratum,
        head_names=EF4_HEADS,
        preregistration_sha=PREREGISTRATION_SHA,
    )
    report = run_extended_fali_seed(cfg, _REPO)
    return 0 if report.get("overall_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
