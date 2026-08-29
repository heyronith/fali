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
FIRST_DRAFT = "09cd1b651601dc3588aa6107fe1eb65b59d4a5a6"


@pytest.fixture(scope="module")
def rendered_figures():
    cmd = [sys.executable, str(SCRIPTS / "render_all.py")]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "UserWarning" not in proc.stderr, proc.stderr
    return FIGURES


def _git_show_provenance(stem: str) -> dict:
    rel = f"manuscript/machine_learning/figures/{stem}_provenance.json"
    raw = subprocess.check_output(["git", "show", f"{FIRST_DRAFT}:{rel}"], cwd=REPO, text=True)
    return json.loads(raw)


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


def test_figure2_partition_schematic():
    sys.path.insert(0, str(SCRIPTS))
    from score_partition import build_score_partition, validate_illustrative_partition

    summary = validate_illustrative_partition()
    grid = build_score_partition()
    assert summary.n_T == 1
    assert summary.n_R == summary.q_star == 5
    assert summary.n_O == 15
    assert summary.n_mask == 15
    assert summary.n_valid == 21
    assert grid[(5, 5)] == "R"
    assert summary.final_row == ["R", "R", "T", "R", "R", "R"]
    for q in range(5):
        for k in range(q + 1):
            assert grid[(q, k)] == "O"
    for q in range(6):
        for k in range(6):
            if k > q:
                assert grid[(q, k)] == "MASK"


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


def test_fig5_has_no_unicode_checkmarks():
    text = (SCRIPTS / "render_fig5.py").read_text()
    assert "✓" not in text
    assert "✗" not in text


def test_empirical_extracted_values_unchanged(rendered_figures):
    for stem in ("Fig3", "Fig4", "Fig5"):
        current = json.loads((rendered_figures / f"{stem}_provenance.json").read_text())
        baseline = _git_show_provenance(stem)
        assert current["extracted_values"] == baseline["extracted_values"], stem


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
        if n >= 2:
            assert doc.get("revision_note") == "visual/scientific-diagram correction only; no empirical recomputation"


def test_contact_sheets(rendered_figures):
    for name in ("figures_contact_sheet.png", "figures_contact_sheet_printsize.png"):
        path = rendered_figures / name
        assert path.exists()
        assert path.stat().st_size > 0


def test_figure_include_files_exist():
    for n in range(1, 6):
        path = FIGURES / f"fig{n}.tex"
        assert path.exists(), path
        text = path.read_text()
        assert f"figures/Fig{n}.pdf" in text
        assert "\\begin{figure*}" in text


def test_manuscript_integrates_all_figures():
    main = (REPO / "manuscript/machine_learning/main.tex").read_text()
    method = (REPO / "manuscript/machine_learning/method_validation.tex").read_text()
    gpt2 = (REPO / "manuscript/machine_learning/gpt2_ioi.tex").read_text()
    pythia = (REPO / "manuscript/machine_learning/pythia_induction.tex").read_text()
    assert "\\input{figures/fig1}" in main
    assert "\\input{figures/fig2}" in main
    assert "\\input{figures/fig3}" in method
    assert "\\input{figures/fig4}" in gpt2
    assert "\\input{figures/fig5}" in pythia
    assert main.count("\\input{figures/fig") == 2


def test_captions_tex_exists():
    path = FIGURES / "captions.tex"
    assert path.exists()
    text = path.read_text()
    assert text.count("\\caption") == 5
    assert "\\label{fig:" in text


def test_fig2_provenance_partition_counts(rendered_figures):
    doc = json.loads((rendered_figures / "Fig2_provenance.json").read_text())
    counts = doc["extracted_values"]["partition_counts"]
    assert counts == {"T": 1, "R": 5, "O": 15, "MASK": 15}
    assert doc["extracted_values"]["q_star_q_star_player"] == "R"


def test_fig3_provenance_sources(rendered_figures):
    doc = json.loads((rendered_figures / "Fig3_provenance.json").read_text())
    paths = [entry["path"] for entry in doc["source_files"]]
    assert "results/extended_fali/ef1_synthetic/validation_report.json" in paths
    assert doc["extracted_values"]["overall_pass"] is True


def test_fig5_gate_status_from_source(rendered_figures):
    doc = json.loads((rendered_figures / "Fig5_provenance.json").read_text())
    assert doc["extracted_values"]["ef6_gate"] == "FAIL"
    assert doc["extracted_values"]["ef6_gates"]["G3"] == "NOT_CONFIRMED"
