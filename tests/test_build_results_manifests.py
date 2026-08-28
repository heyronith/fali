"""Tests for tools/build_results_manifests.py."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "build_results_manifests.py"
RESULTS = REPO / "results"


def test_build_and_validate_manifests():
    proc = subprocess.run(
        [sys.executable, str(TOOL)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "Post-build validation PASS" in proc.stdout

    validate = subprocess.run(
        [sys.executable, str(TOOL), "--validate"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert validate.returncode == 0, validate.stderr or validate.stdout
    assert "VALIDATION PASS" in validate.stdout


def test_manifest_excludes_self_entries():
    raw = json.loads((RESULTS / "RAW_DATA_INDEX.json").read_text())
    hist = json.loads((RESULTS / "HISTORICAL_RESULTS_MANIFEST.json").read_text())
    for doc in (raw, hist):
        for entry in doc["files"]:
            name = Path(entry["relative_path"]).name
            assert name not in {
                "HISTORICAL_RESULTS_MANIFEST.json",
                "RAW_DATA_INDEX.json",
            }


def test_revision_seed_entries_have_config_path():
    raw = json.loads((RESULTS / "RAW_DATA_INDEX.json").read_text())
    for entry in raw["files"]:
        rel = entry["relative_path"]
        if "/full/seed_" in rel and rel.startswith("results/revision_phase"):
            assert entry.get("config_path"), f"missing config_path: {rel}"
            assert (REPO / entry["config_path"]).exists()


def test_git_tracked_is_boolean():
    raw = json.loads((RESULTS / "RAW_DATA_INDEX.json").read_text())
    for entry in raw["files"]:
        assert isinstance(entry["git_tracked"], bool)


def test_unknown_generator_notes():
    raw = json.loads((RESULTS / "RAW_DATA_INDEX.json").read_text())
    for entry in raw["files"]:
        if entry.get("generating_script") is None and entry.get("config_path") is None:
            notes = entry.get("notes", "")
            assert "Regenerate via generating_script" not in notes
            assert notes == "generator/config unknown" or "likely produced" in notes


def test_csv_row_counts_match_disk():
    raw = json.loads((RESULTS / "RAW_DATA_INDEX.json").read_text())
    for entry in raw["files"]:
        rel = entry["relative_path"]
        if not rel.endswith(".csv"):
            continue
        path = REPO / rel
        with open(path, newline="") as f:
            rows = sum(1 for _ in f) - 1
        if entry.get("row_count") is not None:
            assert rows == entry["row_count"], rel
