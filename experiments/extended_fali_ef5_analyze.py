#!/usr/bin/env python3
"""EF-5 aggregate analysis after all schemes and seeds complete."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from extended_fali_ef4_analysis import EF4_SEEDS  # noqa: E402
from extended_fali_ef5_analysis import ALT_SCHEMES, run_analysis  # noqa: E402

PREREGISTRATION_SHA = "835b031db76b47022c74e796ab9e5d6f8d16f40b"


def build_validation_summary(repo: Path, aggregate_dir: Path) -> dict:
    per_scheme_seed = {}
    overall = True
    total_cells = 0
    total_attr = 0
    maxima = {
        "max_score_pattern_error": 0.0,
        "max_z_reconstruction_error": 0.0,
        "max_endpoint_error": 0.0,
        "max_owen_residual": 0.0,
        "max_quotient_consistency_error": 0.0,
    }
    for scheme in ALT_SCHEMES:
        per_scheme_seed[scheme] = {}
        for seed in EF4_SEEDS:
            seed_dir = repo / f"results/extended_fali/ef5_ablation_robustness/{scheme}/seed_{seed}"
            vr = json.loads((seed_dir / "validation_report.json").read_text())
            per_scheme_seed[scheme][str(seed)] = vr.get("overall_pass")
            overall &= bool(vr.get("overall_pass"))
            total_cells += int(vr.get("n_cell_rows", 0))
            total_attr += int(vr.get("n_attribution_rows", 0))
            for k in maxima:
                if k in vr:
                    maxima[k] = max(maxima[k], float(vr[k]))

    summary = {
        "overall_pass": overall,
        "per_scheme_seed": per_scheme_seed,
        **maxima,
        "expected_total_cell_rows": 921600,
        "expected_total_attribution_rows": 28800,
        "observed_total_cell_rows": total_cells,
        "observed_total_attribution_rows": total_attr,
        "preregistration_sha": PREREGISTRATION_SHA,
    }
    if total_cells != 921600 or total_attr != 28800:
        summary["overall_pass"] = False
        summary["row_count_error"] = True
    return summary


def main() -> int:
    aggregate_dir = _REPO / "results/extended_fali/ef5_ablation_robustness/aggregate"
    val = build_validation_summary(_REPO, aggregate_dir)
    results = run_analysis(_REPO, aggregate_dir, preregistration_sha=PREREGISTRATION_SHA)
    val["historical_integrity"] = results["integrity"]
    val["zero_reference_informative"] = results["damage_strength"][0]["INFORMATIVE"]
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    (aggregate_dir / "validation_summary.json").write_text(json.dumps(val, indent=2) + "\n")
    print(json.dumps({"validation": val, "gate": results["gate"]}, indent=2))
    return 0 if val["overall_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
