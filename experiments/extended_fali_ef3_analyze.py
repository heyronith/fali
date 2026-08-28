#!/usr/bin/env python3
"""EF-3 aggregate analysis after all four seeds complete."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from extended_fali_ef3_analysis import EF3_SEEDS, run_analysis  # noqa: E402


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_validation_summary(repo: Path, aggregate_dir: Path) -> dict:
    per_seed = {}
    overall = True
    maxima = {
        "max_score_pattern_error": 0.0,
        "max_z_reconstruction_error": 0.0,
        "max_endpoint_error": 0.0,
        "max_owen_residual": 0.0,
        "max_quotient_consistency_error": 0.0,
        "max_historical_h_discrepancy": 0.0,
        "max_historical_shapley_discrepancy": 0.0,
    }
    raw_index = []
    for seed in EF3_SEEDS:
        seed_dir = repo / f"results/extended_fali/ef3_two_template/seed_{seed}"
        vr_path = seed_dir / "validation_report.json"
        vr = json.loads(vr_path.read_text())
        per_seed[str(seed)] = vr
        overall &= bool(vr.get("overall_pass", False))
        for k in maxima:
            if k in vr:
                maxima[k] = max(maxima[k], float(vr[k]))
        cell_path = seed_dir / "cell_raw.csv.gz"
        if not cell_path.exists():
            cell_path = seed_dir / "cell_raw.csv"
        raw_index.append(
            {
                "seed": seed,
                "path": str(cell_path.relative_to(repo)),
                "sha256": file_sha256(cell_path),
                "row_count": int(vr.get("n_cell_rows", 0)),
                "schema": "seed,example_idx,head_name,layer,head,T,R,O,V,C,metric_injected,metric_zero,U",
            }
        )

    summary = {
        "overall_pass": overall,
        "per_seed": {k: v.get("overall_pass") for k, v in per_seed.items()},
        "dataset_alignment": {k: v.get("checks", {}).get("dataset_alignment") for k, v in per_seed.items()},
        **maxima,
        "expected_total_cell_rows": 307200,
        "expected_total_attribution_rows": 9600,
        "observed_total_cell_rows": sum(r["row_count"] for r in raw_index),
        "raw_cell_index": raw_index,
    }
    (aggregate_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    aggregate_dir = _REPO / "results/extended_fali/ef3_two_template/aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    val = build_validation_summary(_REPO, aggregate_dir)
    results = run_analysis(_REPO, aggregate_dir)
    print(json.dumps({"validation": val, "go_stop": results["go_stop"]}, indent=2))
    return 0 if val["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
