"""Render all publication figures to candidate_final/."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[2]
CANDIDATE_DIR = SCRIPTS.parent / "figures/candidate_final"
FIGURES_DIR = SCRIPTS.parent / "figures"


def _load_render(name: str):
    path = SCRIPTS / f"render_{name}.py"
    spec = importlib.util.spec_from_file_location(f"pub_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.render


def render_all() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS.parent / "figure_scripts/validate_figure_inputs.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + proc.stderr)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    for n in range(1, 6):
        _load_render(f"fig{n}")()
    print(f"Rendered to {CANDIDATE_DIR}")


def install_to_manuscript() -> None:
    for n in range(1, 6):
        for ext in ("png", "svg"):
            shutil.copy2(CANDIDATE_DIR / f"Fig{n}.{ext}", FIGURES_DIR / f"Fig{n}.{ext}")
        prov = CANDIDATE_DIR / f"Fig{n}_provenance.json"
        if prov.exists():
            shutil.copy2(prov, FIGURES_DIR / prov.name)
    print(f"Installed PNG/SVG to {FIGURES_DIR}")


if __name__ == "__main__":
    render_all()
    install_to_manuscript()
