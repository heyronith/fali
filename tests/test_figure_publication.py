"""Tests for publication figure pipeline (PNG manuscript figures)."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
PUB = REPO / "manuscript/machine_learning/figure_scripts_pub"
FIGURES = REPO / "manuscript/machine_learning/figures"
CANDIDATE = FIGURES / "candidate_final"
SCRIPTS = REPO / "manuscript/machine_learning/figure_scripts"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
TEXTWIDTH_IN = json.loads((PUB / "textwidth.json").read_text())["textwidth_in"]
MIN_PX_WIDTH = int(TEXTWIDTH_IN * 300)


def _load_pub_render(name: str):
    path = PUB / f"render_{name}.py"
    spec = importlib.util.spec_from_file_location(f"pub_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pub_rendered():
    """Render to candidate_final only (does not overwrite manuscript PNGs)."""
    proc = subprocess.run(
        [sys.executable, "-c", f"""
import importlib.util, sys
from pathlib import Path
pub = Path({str(PUB)!r})
spec = importlib.util.spec_from_file_location('ra', pub / 'render_all.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.render_all()
"""],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return CANDIDATE


def test_png_magic_bytes():
    for n in range(1, 6):
        data = (FIGURES / f"Fig{n}.png").read_bytes()[:8]
        assert data == PNG_MAGIC, f"Fig{n}.png invalid PNG header"


def test_png_decodes_with_pillow():
    for n in range(1, 6):
        im = Image.open(FIGURES / f"Fig{n}.png")
        im.load()
        assert im.width >= MIN_PX_WIDTH, f"Fig{n} width {im.width} < {MIN_PX_WIDTH}"


def test_wrappers_reference_png_only():
    for n in range(1, 6):
        text = (FIGURES / f"fig{n}.tex").read_text()
        assert f"figures/Fig{n}.png" in text
        assert f"figures/Fig{n}.pdf" not in text


def test_no_hardcoded_empirical_in_pub_scripts():
    locked = [
        r"0\.7828059418",
        r"0\.2984514048",
        r"-0\.1437365829",
        r"0\.4856525427",
    ]
    for script in PUB.glob("render_fig*.py"):
        text = script.read_text()
        for pattern in locked:
            assert re.search(pattern, text) is None, f"{script.name}: {pattern}"


def test_minimum_font_sizes_pub_scripts():
    pattern = re.compile(
        r"fontsize\s*=\s*(6(?:\.5)?|7(?:\.5)?)\b|set_fontsize\s*\(\s*(6(?:\.5)?|7(?:\.5)?)\s*\)"
    )
    files = [PUB / "figure_style.py", *sorted(PUB.glob("render_fig*.py"))]
    offenders = []
    for path in files:
        for m in pattern.finditer(path.read_text()):
            offenders.append(f"{path.name}: {m.group(0)}")
    assert not offenders


def test_scientific_gates(pub_rendered):
    sys.path.insert(0, str(SCRIPTS))
    from figure_data import validate_scientific_inputs

    s = validate_scientific_inputs()
    assert s["ef4_gate"] == "PASS"
    assert s["ef6_gate"] == "FAIL"
    assert s["l10h2"]["conditional_gain"]["point"] > 0
    assert s["l10h2"]["phi_T"]["point"] > 0
    assert s["l10h2"]["phi_R"]["point"] < 0
    assert s["l10h10"]["delta_target_attn"]["point"] > 0
    assert s["l10h10"]["conditional_gain"]["point"] < 0
    assert s["l10h10"]["phi_T"]["point"] > 0
    assert s["l10h10"]["phi_R"]["point"] < 0
    assert s["pythia"]["conditional_gain"]["point"] > 0
    assert s["pythia"]["phi_T"]["point"] < 0
    assert s["pythia"]["phi_R"]["point"] > 0


def test_fig3_provenance_empirical(pub_rendered):
    doc = json.loads((pub_rendered / "Fig3_provenance.json").read_text())
    delta_us = doc["extracted_values"]["single_channel_delta_u"]
    assert len(delta_us) == 5
    assert doc["extracted_values"]["overall_pass"] is True


def test_fig4_provenance_empirical(pub_rendered):
    doc = json.loads((pub_rendered / "Fig4_provenance.json").read_text())
    h2 = doc["extracted_values"]["heads"]["L10H2"]
    assert h2["phi_T"]["point"] > 0
    assert h2["phi_R"]["point"] < 0
    assert doc["extracted_values"]["gate"] == "PASS"


def test_fig5_provenance_empirical(pub_rendered):
    doc = json.loads((pub_rendered / "Fig5_provenance.json").read_text())
    assert doc["extracted_values"]["ef6_gate"] == "FAIL"
    assert doc["extracted_values"]["ef6_gates"]["G3"] == "NOT_CONFIRMED"


def test_figure2_placement_in_manuscript():
    main = (REPO / "manuscript/machine_learning/main.tex").read_text()
    owen_pos = main.index("\\subsection{Standard Owen attribution")
    fig2_pos = main.index("\\input{figures/fig2}")
    assert fig2_pos > owen_pos


def test_figure1_leadin_text():
    main = (REPO / "manuscript/machine_learning/main.tex").read_text()
    assert "cross-state change $\\Delta_h$ in its output-grounded causal usefulness" in main


def test_pub_render_outputs(pub_rendered):
    for n in range(1, 6):
        for ext in ("png", "svg"):
            p = pub_rendered / f"Fig{n}.{ext}"
            assert p.exists() and p.stat().st_size > 0
