"""Tests for FALI main figure rendering pipeline."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "manuscript/machine_learning/figure_scripts"
FIGURES = REPO / "manuscript/machine_learning/figures"


@pytest.fixture(scope="module")
def rendered_figures():
    cmd = [sys.executable, str(SCRIPTS / "render_all.py")]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return FIGURES


def test_required_source_files_exist():
    sys.path.insert(0, str(SCRIPTS))
    from figure_data import assert_paths_exist

    assert_paths_exist()


def test_scientific_validation():
    sys.path.insert(0, str(SCRIPTS))
    from figure_data import validate_scientific_inputs

    summary = validate_scientific_inputs()
    assert summary["ef1_overall_pass"] is True
    assert summary["ef4_gate"] == "PASS"
    assert summary["ef6_gate"] == "FAIL"
    assert summary["l10h2"]["phi_T"]["point"] > 0
    assert summary["l10h2"]["phi_R"]["point"] < 0
    assert summary["pythia"]["phi_T"]["point"] < 0
    assert summary["pythia"]["phi_R"]["point"] > 0


def test_ef4_heads_present():
    sys.path.insert(0, str(SCRIPTS))
    from figure_data import get_ef4_metric, load_ef4_ci

    df = load_ef4_ci()
    for head in ("L10H2", "L10H10"):
        for metric in ("conditional_gain", "delta_target_attn", "phi_T", "phi_R", "Phi_SCORE", "phi_C"):
            get_ef4_metric(df, head, metric)


def test_no_hardcoded_locked_values_in_render_scripts():
    locked = [
        r"0\.7828059418444279",
        r"0\.29845140483644267",
        r"-0\.14373658294868089",
        r"0\.48565254271589864",
    ]
    for script in SCRIPTS.glob("render_fig*.py"):
        text = script.read_text()
        for pattern in locked:
            assert re.search(pattern, text) is None, f"{script.name} contains locked value {pattern}"


def test_figure_outputs_and_provenance(rendered_figures):
    for n in range(1, 6):
        stem = f"Fig{n}"
        for ext in ("pdf", "svg", "png"):
            path = rendered_figures / f"{stem}.{ext}"
            assert path.exists(), path
            assert path.stat().st_size > 0, path
        prov = rendered_figures / f"{stem}_provenance.json"
        assert prov.exists()
        doc = json.loads(prov.read_text())
        assert doc["figure"] == stem
        assert "git_commit" in doc
        assert "plotting_library_versions" in doc


def test_contact_sheet(rendered_figures):
    path = rendered_figures / "figures_contact_sheet.png"
    assert path.exists()
    assert path.stat().st_size > 0


def test_captions_tex_exists():
    path = FIGURES / "captions.tex"
    assert path.exists()
    text = path.read_text()
    assert text.count("\\caption") == 5
    assert "\\label{fig:" in text

def test_fig3_provenance_sources(rendered_figures):
    doc = json.loads((rendered_figures / "Fig3_provenance.json").read_text())
    paths = [entry["path"] for entry in doc["source_files"]]
    assert "results/extended_fali/ef1_synthetic/validation_report.json" in paths
    assert doc["extracted_values"]["overall_pass"] is True


def test_fig5_gate_status_from_source(rendered_figures):
    doc = json.loads((rendered_figures / "Fig5_provenance.json").read_text())
    assert doc["extracted_values"]["ef6_gate"] == "FAIL"
    assert doc["extracted_values"]["ef6_gates"]["G3"] == "NOT_CONFIRMED"
