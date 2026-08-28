#!/usr/bin/env python3
"""Build RAW_DATA_INDEX.json and HISTORICAL_RESULTS_MANIFEST.json for the repo baseline."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

# Files above this size (bytes) are indexed but not intended for Git tracking.
LARGE_RAW_THRESHOLD = 2 * 1024 * 1024

# Glob patterns for large raw outputs (relative to results/).
LARGE_RAW_GLOBS = [
    "raw/*.csv",
    "raw/*.json",
    "revision_phase2/full/aggregate/phase1_phase2_merged_raw.csv",
    "revision_phase2/full/seed_*/phase2_combined_raw.csv",
    "revision_phase2/full/seed_*/fali_raw.csv",
    "revision_phase2/full/seed_*/score_baseline_raw.csv",
    "revision_phase2/full/seed_*/causal_score_hybrid_raw.csv",
    "revision_phase2/full/seed_*/pattern_patch_raw.csv",
    "revision_phase2/full/seed_*/qk_patch_raw.csv",
    "revision_phase2/full/seed_*/qk_score_decomposition_raw.csv",
    "revision_phase3c/full/seed_*/factorial_raw.csv",
    "revision_phase3c/full/aggregate/factorial_merged_raw.csv",
    "revision_phase4/full/seed_*/factorial_raw.csv",
]

SCRIPT_MAP = {
    "revision_phase1": "experiments/revision_phase1_conditional_gain.py",
    "revision_phase2": "experiments/revision_phase2_fali_causal.py",
    "revision_phase3c": "experiments/revision_phase3c_factorial.py",
    "revision_phase4": "experiments/revision_phase4_factorial.py",
}

CONFIG_HINTS = {
    "revision_phase1/full/seed_": "results/revision_phase1/full/seed_{seed}/run_config.json",
    "revision_phase2/full/seed_": "results/revision_phase2/full/seed_{seed}/run_config.json",
    "revision_phase3c/full/seed_": "results/revision_phase3c/full/seed_{seed}/run_config.json",
    "revision_phase4/full/seed_": "results/revision_phase4/full/seed_{seed}/run_config.json",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_count_csv(path: Path) -> int | None:
    try:
        with open(path, newline="") as f:
            return sum(1 for _ in f) - 1  # minus header
    except Exception:
        return None


def schema_csv(path: Path) -> list[str] | None:
    try:
        with open(path, newline="") as f:
            return next(csv.reader(f))
    except Exception:
        return None


def infer_experiment(rel: str) -> str:
    if rel.startswith("raw/"):
        return "original_fali"
    for key in ("revision_phase1", "revision_phase2", "revision_phase3c", "revision_phase4"):
        if key in rel:
            return key
    if rel.startswith("summary/"):
        return "original_summary"
    return "unknown"


def infer_script(rel: str) -> str | None:
    exp = infer_experiment(rel)
    return SCRIPT_MAP.get(exp)


def infer_seeds(rel: str) -> list[int] | str | None:
    if "/seed_42/" in rel:
        return [42]
    if "/seed_1/" in rel:
        return [1]
    if "/seed_8/" in rel:
        return [8]
    if "/seed_22/" in rel:
        return [22]
    if "full/aggregate" in rel or "full/seed_" in rel:
        return [42, 1, 8, 22]
    return None


def collect_large_raw_set() -> set[str]:
    out: set[str] = set()
    for pat in LARGE_RAW_GLOBS:
        for p in RESULTS.glob(pat):
            if p.is_file():
                out.add(str(p.relative_to(RESULTS)).replace("\\", "/"))
    return out


def is_scientific_result(path: Path) -> bool:
    rel = str(path.relative_to(RESULTS)).replace("\\", "/")
    skip_parts = (
        "__pycache__",
        ".DS_Store",
    )
    if any(x in rel for x in skip_parts):
        return False
    return True


def main() -> None:
    large_set = collect_large_raw_set()
    raw_index = []
    manifest = []

    for path in sorted(RESULTS.rglob("*")):
        if not path.is_file() or not is_scientific_result(path):
            continue
        rel = str(path.relative_to(RESULTS)).replace("\\", "/")
        size = path.stat().st_size
        entry = {
            "relative_path": f"results/{rel}",
            "size_bytes": size,
            "sha256": sha256_file(path),
        }
        manifest.append(entry)

        is_large = rel in large_set or size > LARGE_RAW_THRESHOLD
        if not is_large:
            continue
        if rel.endswith(".csv"):
            rc = row_count_csv(path)
            schema = schema_csv(path)
        else:
            rc, schema = None, None

        raw_index.append(
            {
                "experiment": infer_experiment(rel),
                "relative_path": f"results/{rel}",
                "row_count": rc,
                "schema": schema,
                "size_bytes": size,
                "sha256": entry["sha256"],
                "generating_script": infer_script(rel),
                "config_path": None,
                "seeds": infer_seeds(rel),
                "git_tracked": False,
                "notes": (
                    "Large per-example raw output; indexed for reproducibility. "
                    "Regenerate via generating_script + run_config.json."
                ),
            }
        )

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "large_raw_threshold_bytes": LARGE_RAW_THRESHOLD,
        "n_indexed_large_raw_files": len(raw_index),
        "n_manifest_files": len(manifest),
    }
    (RESULTS / "RAW_DATA_INDEX.json").write_text(
        json.dumps({"meta": meta, "files": raw_index}, indent=2) + "\n"
    )
    (RESULTS / "HISTORICAL_RESULTS_MANIFEST.json").write_text(
        json.dumps({"meta": meta, "files": manifest}, indent=2) + "\n"
    )
    print(f"Wrote {len(raw_index)} large raw index entries")
    print(f"Wrote {len(manifest)} manifest entries")


if __name__ == "__main__":
    main()
