#!/usr/bin/env python3
"""EF-4 aggregate analysis after all four canonical seeds complete."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from extended_fali_ef4_analysis import EF4_SEEDS, run_analysis  # noqa: E402

PREREGISTRATION_SHA = "2661459f047f65fabc418993d00e3bb06cc0b3b6"


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
    observed_attribution_rows = 0
    for seed in EF4_SEEDS:
        seed_dir = repo / f"results/extended_fali/ef4_canonical/seed_{seed}"
        vr_path = seed_dir / "validation_report.json"
        vr = json.loads(vr_path.read_text())
        per_seed[str(seed)] = vr
        overall &= bool(vr.get("overall_pass", False))
        n_attr = sum(1 for _ in open(seed_dir / "attribution_raw.csv")) - 1
        observed_attribution_rows += n_attr
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
        "expected_total_cell_rows": 460800,
        "expected_total_attribution_rows": 14400,
        "observed_total_attribution_rows": observed_attribution_rows,
        "observed_total_cell_rows": sum(r["row_count"] for r in raw_index),
        "raw_cell_index": raw_index,
        "preregistration_sha": PREREGISTRATION_SHA,
    }
    if observed_attribution_rows != 14400:
        summary["overall_pass"] = False
        summary["attribution_row_count_error"] = (
            f"expected 14400 attribution rows, observed {observed_attribution_rows}"
        )
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    (aggregate_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    aggregate_dir = _REPO / "results/extended_fali/ef4_canonical/aggregate"
    val = build_validation_summary(_REPO, aggregate_dir)
    results = run_analysis(_REPO, aggregate_dir, preregistration_sha=PREREGISTRATION_SHA)
    print(json.dumps({"validation": val, "fine_score_gate": results["fine_score_gate"]}, indent=2))
    return 0 if val["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
