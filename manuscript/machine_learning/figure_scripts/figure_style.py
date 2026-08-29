"""Shared visual system for FALI main figures."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
FIGURES_DIR = REPO_ROOT / "manuscript/machine_learning/figures"
SCRIPTS_DIR = REPO_ROOT / "manuscript/machine_learning/figure_scripts"

FIG_WIDTH_MM = 174.0
FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4
DPI_PNG = 300

COLORS = {
    "T": "#0072B2",
    "R": "#D55E00",
    "O": "#7F7F7F",
    "V": "#009E73",
    "C": "#CC79A7",
    "endpoint": "#000000",
    "mask": "#CCCCCC",
    "neutral": "#333333",
    "pass": "#009E73",
    "fail": "#666666",
}

MARKERS = {
    "T": "o",
    "R": "s",
    "endpoint": "D",
}

CHANNELS = ["T", "R", "O", "V", "C"]
PHI_COLS = ["phi_T", "phi_R", "phi_O", "phi_V", "phi_C"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNKNOWN"


def library_versions() -> Dict[str, str]:
    import numpy as np
    import pandas as pd

    return {
        "matplotlib": mpl.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.grid": False,
        }
    )


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.08,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_figure(
    fig: plt.Figure,
    stem: str,
    *,
    script_path: Path,
    source_files: Sequence[Path],
    extracted_values: Dict[str, Any],
    tight: bool = True,
    revision_note: Optional[str] = None,
) -> Dict[str, Any]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()

    outputs: Dict[str, str] = {}
    for ext, kwargs in (
        ("pdf", {"format": "pdf", "bbox_inches": "tight", "pad_inches": 0.02}),
        ("svg", {"format": "svg", "bbox_inches": "tight", "pad_inches": 0.02}),
        ("png", {"format": "png", "dpi": DPI_PNG, "bbox_inches": "tight", "pad_inches": 0.02}),
    ):
        out = FIGURES_DIR / f"{stem}.{ext}"
        fig.savefig(out, **kwargs)
        outputs[ext] = str(out.relative_to(REPO_ROOT))

    size = fig.get_size_inches()
    provenance = {
        "figure": stem,
        "generating_script": str(script_path.relative_to(REPO_ROOT)),
        "git_commit": git_commit(),
        "revision_note": revision_note,
        "source_files": [
            {
                "path": str(p.relative_to(REPO_ROOT)),
                "sha256": sha256_file(p),
            }
            for p in source_files
        ],
        "extracted_values": extracted_values,
        "output_dimensions_inches": {"width": float(size[0]), "height": float(size[1])},
        "output_width_mm": FIG_WIDTH_MM,
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "plotting_library_versions": library_versions(),
        "outputs": outputs,
    }
    prov_path = FIGURES_DIR / f"{stem}_provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    plt.close(fig)
    return provenance


def zero_line(ax, axis: str = "x", **kwargs) -> None:
    defaults = {"color": "black", "linewidth": 1.2, "zorder": 1}
    defaults.update(kwargs)
    if axis == "x":
        ax.axvline(0, **defaults)
    else:
        ax.axhline(0, **defaults)


def format_sci(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 1e-3:
        return f"{value:.3g}"
    return f"{value:.2e}"
