#!/usr/bin/env python3
"""Build and validate RAW_DATA_INDEX.json and HISTORICAL_RESULTS_MANIFEST.json."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

MANIFEST_SELF = {
    "HISTORICAL_RESULTS_MANIFEST.json",
    "RAW_DATA_INDEX.json",
}

LARGE_RAW_THRESHOLD = 2 * 1024 * 1024

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
    "extended_fali/ef3_two_template/seed_*/cell_raw.csv.gz",
    "extended_fali/ef3_two_template/aggregate/attribution_merged_raw.csv",
]

REVISION_SCRIPT_MAP = {
    "revision_phase1": "experiments/revision_phase1_conditional_gain.py",
    "revision_phase2": "experiments/revision_phase2_fali_causal.py",
    "revision_phase3c": "experiments/revision_phase3c_factorial.py",
    "revision_phase4": "experiments/revision_phase4_factorial.py",
    "extended_fali_ef3": "experiments/extended_fali_ef3_full.py",
}

# Original-FALI raw filename stem -> generating script (exact where known).
ORIGINAL_RAW_SCRIPT_MAP = {
    "ablation_controls_raw": "experiments/ablation_controls.py",
    "fali_decomposition_raw": "experiments/fali_decomposition.py",
    "qk_decomposition_raw": "experiments/qk_decomposition.py",
    "output_mediation_raw": "experiments/output_mediation.py",
    "fali_raw_results": "experiments/hybrid_intervention.py",
    "ln_frozen_raw_results": "experiments/fali_rigorous_validation.py",
    "attention_results": None,  # aggregated; generator unknown for raw/json export path
}

SEED_RE = re.compile(r"/seed_(\d+)/")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_count_csv(path: Path) -> Optional[int]:
    try:
        with open(path, newline="") as f:
            return sum(1 for _ in f) - 1
    except OSError:
        return None


def schema_csv(path: Path) -> Optional[List[str]]:
    try:
        with open(path, newline="") as f:
            return next(csv.reader(f))
    except (OSError, StopIteration):
        return None


def git_tracked(rel_from_repo: str) -> bool:
    try:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel_from_repo],
            cwd=REPO,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def infer_experiment(rel: str) -> str:
    if rel.startswith("raw/"):
        return "original_fali"
    if "extended_fali/ef3_two_template" in rel:
        return "extended_fali_ef3"
    for key in REVISION_SCRIPT_MAP:
        if key in rel:
            return key
    if rel.startswith("summary/"):
        return "original_summary"
    return "unknown"


def infer_revision_config(rel: str) -> Optional[str]:
    m = SEED_RE.search(f"/{rel}")
    if not m:
        return None
    seed = m.group(1)
    for phase in ("revision_phase1", "revision_phase2", "revision_phase3c", "revision_phase4"):
        if phase in rel:
            return f"results/{phase}/full/seed_{seed}/run_config.json"
    if "extended_fali/ef3_two_template/seed_" in rel:
        return f"results/extended_fali/ef3_two_template/seed_{seed}/run_config.json"
    if rel == "extended_fali/ef3_two_template/aggregate/attribution_merged_raw.csv":
        return "results/extended_fali/ef3_two_template/aggregate/analysis_config.json"
    return None


def infer_original_script(rel: str) -> Optional[str]:
    name = Path(rel).name
    stem = Path(rel).stem
    if stem.endswith("_raw"):
        stem = stem[: -len("_raw")]
    for key, script in ORIGINAL_RAW_SCRIPT_MAP.items():
        if key in name or key in stem:
            return script
    if name == "fali_raw_results.csv" or name == "fali_raw_results.json":
        return ORIGINAL_RAW_SCRIPT_MAP["fali_raw_results"]
    if name.startswith("ln_frozen"):
        return ORIGINAL_RAW_SCRIPT_MAP["ln_frozen_raw_results"]
    return None


def infer_generating_script(rel: str) -> Tuple[Optional[str], str]:
    exp = infer_experiment(rel)
    if exp in REVISION_SCRIPT_MAP:
        return REVISION_SCRIPT_MAP[exp], "revision_phase_script"
    script = infer_original_script(rel)
    if script:
        return script, "original_raw_map"
    if rel.startswith("summary/"):
        return None, "summary_aggregate"
    return None, "unknown"


def infer_config_path(rel: str) -> Optional[str]:
    cfg = infer_revision_config(rel)
    if cfg:
        return cfg
    return None


def infer_seeds(rel: str) -> Optional[List[int]]:
    m = SEED_RE.search(f"/{rel}")
    if m:
        return [int(m.group(1))]
    if "full/aggregate" in rel:
        return [42, 1, 8, 22]
    return None


def build_notes(
    *,
    rel: str,
    script: Optional[str],
    config: Optional[str],
) -> str:
    if script and config:
        return (
            "Large per-example raw output; indexed for reproducibility. "
            f"Regenerate with {script} using {config}."
        )
    if script and not config:
        if infer_experiment(rel) == "original_fali":
            return (
                "Original-FALI raw output; indexed for reproducibility. "
                f"Likely produced by {script}; exact run config not recorded in repo."
            )
        return f"Indexed raw output; likely produced by {script}; config_path unknown."
    return "generator/config unknown"


def collect_large_raw_set() -> Set[str]:
    out: Set[str] = set()
    for pat in LARGE_RAW_GLOBS:
        for p in RESULTS.glob(pat):
            if p.is_file():
                out.add(str(p.relative_to(RESULTS)).replace("\\", "/"))
    return out


def should_scan(path: Path) -> bool:
    if not path.is_file():
        return False
    rel = path.relative_to(RESULTS).as_posix()
    if rel in MANIFEST_SELF:
        return False
    if "__pycache__" in rel or rel.endswith(".DS_Store"):
        return False
    return True


def build_manifests() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    large_set = collect_large_raw_set()
    raw_index: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []

    for path in sorted(RESULTS.rglob("*")):
        if not should_scan(path):
            continue
        rel = path.relative_to(RESULTS).as_posix()
        repo_rel = f"results/{rel}"
        size = path.stat().st_size
        digest = sha256_file(path)
        manifest.append(
            {
                "relative_path": repo_rel,
                "size_bytes": size,
                "sha256": digest,
            }
        )

        is_large = rel in large_set or size > LARGE_RAW_THRESHOLD
        if not is_large:
            continue

        script, _ = infer_generating_script(rel)
        config = infer_config_path(rel)
        if script and not Path(REPO / script).exists():
            script = None
        if config and not Path(REPO / config).exists():
            config = None

        rc = schema = None
        if rel.endswith(".csv"):
            rc = row_count_csv(path)
            schema = schema_csv(path)

        notes = build_notes(rel=rel, script=script, config=config)

        raw_index.append(
            {
                "experiment": infer_experiment(rel),
                "relative_path": repo_rel,
                "row_count": rc,
                "schema": schema,
                "size_bytes": size,
                "sha256": digest,
                "generating_script": script,
                "config_path": config,
                "seeds": infer_seeds(rel),
                "git_tracked": git_tracked(repo_rel),
                "notes": notes,
            }
        )

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "large_raw_threshold_bytes": LARGE_RAW_THRESHOLD,
        "n_indexed_large_raw_files": len(raw_index),
        "n_manifest_files": len(manifest),
    }
    return {"meta": meta, "files": raw_index}, {"meta": meta, "files": manifest}


def write_manifests() -> None:
    raw_doc, hist_doc = build_manifests()
    (RESULTS / "RAW_DATA_INDEX.json").write_text(json.dumps(raw_doc, indent=2) + "\n")
    (RESULTS / "HISTORICAL_RESULTS_MANIFEST.json").write_text(
        json.dumps(hist_doc, indent=2) + "\n"
    )
    print(f"Wrote {len(raw_doc['files'])} large raw index entries")
    print(f"Wrote {len(hist_doc['files'])} manifest entries")


def validate_manifests() -> List[str]:
    errors: List[str] = []
    raw_path = RESULTS / "RAW_DATA_INDEX.json"
    hist_path = RESULTS / "HISTORICAL_RESULTS_MANIFEST.json"
    if not raw_path.exists() or not hist_path.exists():
        return ["Missing RAW_DATA_INDEX.json or HISTORICAL_RESULTS_MANIFEST.json"]

    raw_doc = json.loads(raw_path.read_text())
    hist_doc = json.loads(hist_path.read_text())

    for name in MANIFEST_SELF:
        for doc, label in ((raw_doc, "RAW"), (hist_doc, "HIST")):
            for entry in doc.get("files", []):
                rp = entry.get("relative_path", "")
                if rp.endswith(name) or rp.endswith(f"results/{name}"):
                    errors.append(f"{label} index contains self-entry: {rp}")

    for entry in raw_doc.get("files", []):
        rel = entry["relative_path"]
        path = REPO / rel
        if not path.exists():
            errors.append(f"Indexed file missing: {rel}")
            continue
        if path.stat().st_size != entry["size_bytes"]:
            errors.append(f"Size mismatch: {rel}")
        if sha256_file(path) != entry["sha256"]:
            errors.append(f"SHA256 mismatch: {rel}")
        if rel.endswith(".csv"):
            rc = row_count_csv(path)
            if entry.get("row_count") is not None and rc != entry["row_count"]:
                errors.append(f"Row count mismatch: {rel} ({rc} vs {entry['row_count']})")
            sch = schema_csv(path)
            if entry.get("schema") is not None and sch != entry["schema"]:
                errors.append(f"Schema mismatch: {rel}")
        script = entry.get("generating_script")
        if script and not (REPO / script).exists():
            errors.append(f"generating_script missing: {script} for {rel}")
        cfg = entry.get("config_path")
        if cfg and not (REPO / cfg).exists():
            errors.append(f"config_path missing: {cfg} for {rel}")
        notes = entry.get("notes", "")
        if script is None and cfg is None and "Regenerate via generating_script" in notes:
            errors.append(f"Misleading regeneration note without script/config: {rel}")

    for entry in hist_doc.get("files", []):
        rel = entry["relative_path"]
        path = REPO / rel
        if not path.exists():
            errors.append(f"Manifest file missing: {rel}")
            continue
        if sha256_file(path) != entry["sha256"]:
            errors.append(f"Manifest SHA256 mismatch: {rel}")

    return errors


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing manifest/index files without rewriting.",
    )
    args = p.parse_args(argv)
    if args.validate:
        errors = validate_manifests()
        if errors:
            print("VALIDATION FAIL", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 1
        print("VALIDATION PASS")
        return 0
    write_manifests()
    errors = validate_manifests()
    if errors:
        print("Post-build validation FAIL", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1
    print("Post-build validation PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
