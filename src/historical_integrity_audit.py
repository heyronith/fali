"""
Non-tautological historical artifact integrity checks for Extended-FALI phases.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from extended_fali_ef4_analysis import EF4_SEEDS


def file_sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha256(repo: Path, git_ref: str, rel_path: str) -> Optional[str]:
    try:
        blob = subprocess.check_output(
            ["git", "rev-parse", f"{git_ref}:{rel_path}"],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        data = subprocess.check_output(["git", "cat-file", "blob", blob], cwd=str(repo))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return hashlib.sha256(data).hexdigest()


def load_validation_hash_reports(
    repo: Path,
    seed_dirs: Sequence[Path],
) -> Dict:
    per_seed = {}
    for seed_dir in seed_dirs:
        vr_path = seed_dir / "validation_report.json"
        if not vr_path.exists():
            per_seed[str(seed_dir.name)] = {"status": "NOT_AVAILABLE"}
            continue
        vr = json.loads(vr_path.read_text())
        per_seed[str(seed_dir.name)] = {
            "historical_hash_checks": vr.get("historical_hash_checks"),
            "historical_hashes_before": vr.get("historical_hashes_before"),
            "historical_hashes_after": vr.get("historical_hashes_after"),
            "hash_integrity_check_pass": vr.get("checks", {}).get("historical_hash_integrity"),
        }
    return per_seed


def compare_git_blob_across_refs(
    repo: Path,
    rel_paths: Sequence[str],
    ref_a: str,
    ref_b: str,
) -> Dict:
    rows = []
    all_match = True
    for rel in rel_paths:
        ha = git_blob_sha256(repo, ref_a, rel)
        hb = git_blob_sha256(repo, ref_b, rel)
        if ha is None or hb is None:
            status = "NOT_AVAILABLE"
            all_match = False
        elif ha == hb:
            status = "UNCHANGED"
        else:
            status = "CHANGED"
            all_match = False
        rows.append({"path": rel, "ref_a": ref_a, "ref_b": ref_b, "sha256_a": ha, "sha256_b": hb, "status": status})
    return {"all_match": all_match, "comparisons": rows}


def audit_ef5_historical_integrity(
    repo: Path,
    *,
    ef4_base_ref: str = "79feba9d9a97c62ada673159aaa1d13f757da05b",
    ef5_branch_ref: str = "HEAD",
) -> Dict:
    """Aggregate integrity audit using per-seed validation reports and git blob checks."""
    tracked_paths: List[str] = []
    for seed in EF4_SEEDS:
        tracked_paths.extend(
            [
                f"results/extended_fali/ef4_canonical/seed_{seed}/attribution_raw.csv",
                f"results/revision_phase4/full/seed_{seed}/factorial_raw.csv",
            ]
        )
    tracked_paths.extend(
        [
            "results/extended_fali/ef4_canonical/aggregate/hypothesis_results.json",
            "results/extended_fali/ef4_canonical/aggregate/fine_score_gate.json",
        ]
    )

    ef5_seed_dirs = [
        repo / f"results/extended_fali/ef5_ablation_robustness/matched_mean/seed_{s}" for s in EF4_SEEDS
    ] + [
        repo / f"results/extended_fali/ef5_ablation_robustness/matched_resample/seed_{s}" for s in EF4_SEEDS
    ]

    validation_reports = load_validation_hash_reports(repo, ef5_seed_dirs)

    git_compare = compare_git_blob_across_refs(repo, tracked_paths, ef4_base_ref, ef5_branch_ref)

    raw_index_path = repo / "results/RAW_DATA_INDEX.json"
    raw_index_entries = []
    if raw_index_path.exists():
        idx = json.loads(raw_index_path.read_text())
        for entry in idx.get("entries", []):
            rel = entry.get("relative_path", "")
            if "revision_phase4" in rel and "factorial_raw" in rel:
                disk = repo / rel
                raw_index_entries.append(
                    {
                        "relative_path": rel,
                        "index_sha256": entry.get("sha256"),
                        "disk_sha256": file_sha256(disk),
                        "match": entry.get("sha256") == file_sha256(disk) if disk.exists() else False,
                    }
                )

    per_seed_pass = all(
        v.get("hash_integrity_check_pass") is True
        for k, v in validation_reports.items()
        if k.startswith("seed_")
    )
    overall = per_seed_pass and git_compare["all_match"]

    return {
        "overall_pass": overall,
        "method_note": (
            "Uses per-seed validation_report pre/post hashes and git blob equality "
            "between EF-4 base and EF-5 branch; NOT snapshot(self,self)."
        ),
        "ef4_base_ref": ef4_base_ref,
        "ef5_branch_ref": ef5_branch_ref,
        "validation_reports": validation_reports,
        "git_blob_compare": git_compare,
        "raw_data_index_factorial": raw_index_entries,
    }
