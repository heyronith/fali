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

RAW_AUDIT_REF = "audit/raw-baseline"


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


def tracked_ef4_paths() -> List[str]:
    paths: List[str] = []
    for seed in EF4_SEEDS:
        paths.append(f"results/extended_fali/ef4_canonical/seed_{seed}/attribution_raw.csv")
    paths.extend(
        [
            "results/extended_fali/ef4_canonical/aggregate/hypothesis_results.json",
            "results/extended_fali/ef4_canonical/aggregate/fine_score_gate.json",
        ]
    )
    return paths


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
    for rel in rel_paths:
        ha = git_blob_sha256(repo, ref_a, rel)
        hb = git_blob_sha256(repo, ref_b, rel)
        if ha is None or hb is None:
            status = "NOT_AVAILABLE"
        elif ha == hb:
            status = "UNCHANGED"
        else:
            status = "CHANGED"
        rows.append({"path": rel, "ref_a": ref_a, "ref_b": ref_b, "sha256_a": ha, "sha256_b": hb, "status": status})
    return {"comparisons": rows}


def tracked_git_blob_pass(git_compare: Dict) -> bool:
    """True only when every tracked path is UNCHANGED; NOT_AVAILABLE never counts as equality."""
    return all(row["status"] == "UNCHANGED" for row in git_compare["comparisons"])


def audit_untracked_factorial_raw_index(
    repo: Path,
    *,
    audit_ref: str = RAW_AUDIT_REF,
) -> Dict:
    """
    Verify untracked Phase-4 factorial raws against RAW_DATA_INDEX sha256 and audit branch blobs.
    Does not require files to exist in ordinary EF-4/EF-5 Git trees.
    """
    raw_index_path = repo / "results/RAW_DATA_INDEX.json"
    entries: List[Dict] = []
    if not raw_index_path.exists():
        return {
            "pass": False,
            "audit_ref": audit_ref,
            "entries": entries,
            "note": "RAW_DATA_INDEX.json missing",
        }

    idx = json.loads(raw_index_path.read_text())
    all_pass = True
    for entry in idx.get("files", []):
        rel = entry.get("relative_path", "")
        if "revision_phase4" not in rel or "factorial_raw" not in rel:
            continue
        expected = entry.get("sha256")
        audit_sha = git_blob_sha256(repo, audit_ref, rel)
        if expected is None or audit_sha is None:
            status = "NOT_AVAILABLE"
            match = False
            all_pass = False
        elif expected == audit_sha:
            status = "MATCH"
            match = True
        else:
            status = "MISMATCH"
            match = False
            all_pass = False
        entries.append(
            {
                "relative_path": rel,
                "index_sha256": expected,
                "audit_ref_sha256": audit_sha,
                "status": status,
                "match": match,
            }
        )

    return {"pass": all_pass, "audit_ref": audit_ref, "entries": entries}


def per_seed_prepost_pass(validation_reports: Dict) -> bool:
    for key, report in validation_reports.items():
        if not key.startswith("seed_"):
            continue
        if report.get("status") == "NOT_AVAILABLE":
            return False
        if report.get("hash_integrity_check_pass") is not True:
            return False
    return True


def audit_ef5_historical_integrity(
    repo: Path,
    *,
    ef4_base_ref: str = "79feba9d9a97c62ada673159aaa1d13f757da05b",
    ef5_branch_ref: str = "HEAD",
    raw_audit_ref: str = RAW_AUDIT_REF,
) -> Dict:
    """Aggregate integrity audit using per-seed validation reports, tracked git blobs, and raw index."""
    ef5_seed_dirs = [
        repo / f"results/extended_fali/ef5_ablation_robustness/matched_mean/seed_{s}" for s in EF4_SEEDS
    ] + [
        repo / f"results/extended_fali/ef5_ablation_robustness/matched_resample/seed_{s}" for s in EF4_SEEDS
    ]

    validation_reports = load_validation_hash_reports(repo, ef5_seed_dirs)
    git_compare = compare_git_blob_across_refs(
        repo, tracked_ef4_paths(), ef4_base_ref, ef5_branch_ref
    )
    raw_index_audit = audit_untracked_factorial_raw_index(repo, audit_ref=raw_audit_ref)

    tracked_pass = tracked_git_blob_pass(git_compare)
    untracked_pass = raw_index_audit["pass"]
    prepost_pass = per_seed_prepost_pass(validation_reports)
    overall = tracked_pass and untracked_pass and prepost_pass

    return {
        "overall_pass": overall,
        "tracked_git_blob_pass": tracked_pass,
        "untracked_raw_index_audit_pass": untracked_pass,
        "per_seed_prepost_pass": prepost_pass,
        "method_note": (
            "Tracked EF-4 artifacts: git blob equality between EF-4 base and current branch. "
            "Untracked Phase-4 factorial raws: RAW_DATA_INDEX sha256 vs audit branch blob. "
            "Per-seed EF-5 validation_report pre/post hashes as independent supporting evidence. "
            "NOT snapshot(self,self); NOT_AVAILABLE never counts as equality."
        ),
        "ef4_base_ref": ef4_base_ref,
        "ef5_branch_ref": ef5_branch_ref,
        "raw_audit_ref": raw_audit_ref,
        "validation_reports": validation_reports,
        "git_blob_compare": git_compare,
        "raw_data_index_factorial": raw_index_audit,
    }
