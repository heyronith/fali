#!/usr/bin/env python3
"""EF-5 — one seed × one damage scheme (matched_mean or matched_resample)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from extended_fali_ef5_runner import (  # noqa: E402
    EF5_HEADS,
    EF5RunConfig,
    ef5_required_hash_paths,
    resolve_device,
    run_ef5_seed,
)

EF5_SEEDS = {42, 1, 8, 22}
N_EXPECTED = 900
NUM_PER_STRATUM = 30
PREREGISTRATION_SHA = "835b031db76b47022c74e796ab9e5d6f8d16f40b"
SCHEMES = {"matched_mean", "matched_resample"}


def assert_safe_output_dir(output_dir: Path, repo: Path, scheme: str) -> None:
    rel = str(output_dir.resolve().relative_to(repo.resolve())).replace("\\", "/")
    expected = f"results/extended_fali/ef5_ablation_robustness/{scheme}"
    if expected not in rel:
        raise ValueError(f"Refusing output path: {rel}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scheme", required=True, choices=sorted(SCHEMES))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num-samples", type=int, default=N_EXPECTED)
    p.add_argument("--num-per-stratum", type=int, default=NUM_PER_STRATUM)
    p.add_argument("--device", default="auto")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    if args.seed not in EF5_SEEDS:
        raise SystemExit(f"EF-5 allows seeds {sorted(EF5_SEEDS)} only")
    if args.num_samples != N_EXPECTED:
        raise SystemExit(f"EF-5 requires N={N_EXPECTED}")
    if args.num_per_stratum != NUM_PER_STRATUM:
        raise SystemExit(f"EF-5 requires num_per_stratum={NUM_PER_STRATUM}")

    out = (
        Path(args.output_dir)
        if args.output_dir
        else _REPO / f"results/extended_fali/ef5_ablation_robustness/{args.scheme}/seed_{args.seed}"
    )
    if not out.is_absolute():
        out = _REPO / out
    assert_safe_output_dir(out, _REPO, args.scheme)

    device = resolve_device(args.device)
    if args.require_cuda and not str(device).startswith("cuda"):
        raise RuntimeError(f"--require-cuda but device={device}")

    ref_meta = _REPO / f"results/revision_phase4/full/seed_{args.seed}/dataset_metadata.csv"
    if not ref_meta.exists():
        raise FileNotFoundError(ref_meta)
    for rel in ef5_required_hash_paths(_REPO, args.seed):
        if not (_REPO / rel).exists():
            raise FileNotFoundError(f"Required historical file missing: {rel}")

    cfg = EF5RunConfig(
        phase="EF-5",
        damage_scheme=args.scheme,
        num_samples=args.num_samples,
        seed=args.seed,
        out_dir=out,
        reference_meta=ref_meta,
        required_hash_paths=ef5_required_hash_paths(_REPO, args.seed),
        device=device,
        num_per_stratum=args.num_per_stratum,
        head_names=EF5_HEADS,
        preregistration_sha=PREREGISTRATION_SHA,
    )
    report = run_ef5_seed(cfg, _REPO)
    return 0 if report.get("overall_pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
