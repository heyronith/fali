#!/usr/bin/env python3
"""EF-5 implementation smoke: seed=42, one stratum, 8 examples, both matched schemes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

import pandas as pd  # noqa: E402

from extended_fali_ef5_runner import (  # noqa: E402
    EF5_HEADS,
    EF5RunConfig,
    ef5_required_hash_paths,
    resolve_device,
    run_ef5_seed,
)

PREREGISTRATION_SHA = "835b031db76b47022c74e796ab9e5d6f8d16f40b"
SMOKE_STRATUM = "T00_BABA"
SMOKE_N = 8


def pick_smoke_indices(repo: Path, seed: int = 42) -> list[int]:
    meta = pd.read_csv(repo / f"results/revision_phase4/full/seed_{seed}/dataset_metadata.csv")
    sub = meta[meta["template_stratum"] == SMOKE_STRATUM].sort_values("example_idx")
    if len(sub) < SMOKE_N:
        raise ValueError(f"Stratum {SMOKE_STRATUM} has only {len(sub)} examples")
    return sub["example_idx"].astype(int).tolist()[:SMOKE_N]


def run_scheme(scheme: str, device: str, indices: list[int]) -> dict:
    out = _REPO / f"results/extended_fali/ef5_ablation_robustness/smoke/{scheme}"
    ref = _REPO / "results/revision_phase4/full/seed_42/dataset_metadata.csv"
    cfg = EF5RunConfig(
        phase="EF-5-smoke",
        damage_scheme=scheme,
        num_samples=SMOKE_N,
        seed=42,
        out_dir=out,
        reference_meta=ref,
        required_hash_paths=ef5_required_hash_paths(_REPO, 42),
        device=device,
        head_names=EF5_HEADS,
        preregistration_sha=PREREGISTRATION_SHA,
        example_indices=indices,
        run_smoke_extras=True,
        compress_cells=False,
    )
    return run_ef5_seed(cfg, _REPO)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--require-cuda", action="store_true")
    args = p.parse_args()
    device = resolve_device(args.device)
    if args.require_cuda and not str(device).startswith("cuda"):
        raise RuntimeError(f"--require-cuda but device={device}")

    indices = pick_smoke_indices(_REPO)
    results = {}
    overall = True
    for scheme in ("matched_mean", "matched_resample"):
        print(f"Smoke {scheme}...", flush=True)
        rep = run_scheme(scheme, device, indices)
        results[scheme] = rep
        overall &= bool(rep.get("overall_pass"))

    summary = {"overall_pass": overall, "schemes": {k: v.get("overall_pass") for k, v in results.items()}}
    out_dir = _REPO / "results/extended_fali/ef5_ablation_robustness/smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
