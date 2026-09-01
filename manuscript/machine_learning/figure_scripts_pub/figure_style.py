"""Publication figure style for FALI main figures (PNG + SVG)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
ML_DIR = REPO_ROOT / "manuscript/machine_learning"
SCRIPTS_DIR = Path(__file__).resolve().parent
CANDIDATE_DIR = ML_DIR / "figures/candidate_final"
FIGURES_DIR = ML_DIR / "figures"

_TW = json.loads((SCRIPTS_DIR / "textwidth.json").read_text())
FIG_WIDTH_PT = float(_TW["textwidth_pt"])
FIG_WIDTH_IN = float(_TW["textwidth_in"])
FIG_WIDTH_MM = float(_TW["textwidth_mm"])
DPI_PNG = int(_TW["dpi_png"])

MIN_FONT_PT = 8.0
PANEL_MARKER_PT = 9.0
PANEL_HEADING_PT = 8.5
AXIS_LABEL_PT = 8.5
TICK_PT = 8.0

COLORS = {
    "T": "#4C72B0",
    "R": "#DD8452",
    "O": "#8C8C8C",
    "V": "#55A868",
    "C": "#8172B3",
    "endpoint": "#222222",
    "mask": "#D9D9D9",
    "neutral": "#333333",
    "grid": "#CCCCCC",
}

MARKERS = {"T": "o", "R": "s", "O": "^", "V": "D", "C": "v", "endpoint": "D"}
PHI_COLS = ["phi_T", "phi_R", "phi_O", "phi_V", "phi_C"]
CHANNELS = ["T", "R", "O", "V", "C"]

FIG_HEIGHT_MM = {
    "Fig1": 62.0,
    "Fig2": 98.0,
    "Fig3": 92.0,
    "Fig4": 108.0,
    "Fig5": 78.0,
}


def fig_height_in(stem: str) -> float:
    return FIG_HEIGHT_MM[stem] / 25.4


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


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": AXIS_LABEL_PT,
            "axes.labelsize": AXIS_LABEL_PT,
            "axes.titlesize": PANEL_HEADING_PT,
            "xtick.labelsize": TICK_PT,
            "ytick.labelsize": TICK_PT,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.grid": False,
        }
    )


def panel_heading(ax, marker: str, heading: str) -> None:
    ax.text(
        0.0,
        1.03,
        f"{marker}  {heading}",
        transform=ax.transAxes,
        fontsize=PANEL_MARKER_PT,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def zero_line(ax, axis: str = "x", **kwargs) -> None:
    defaults = {"color": "#333333", "linewidth": 0.9, "zorder": 1}
    defaults.update(kwargs)
    if axis == "x":
        ax.axvline(0, **defaults)
    else:
        ax.axhline(0, **defaults)


def save_figure(
    fig: plt.Figure,
    stem: str,
    *,
    out_dir: Path,
    script_path: Path,
    source_files: Sequence[Path],
    extracted_values: Dict[str, Any],
    provenance: bool = True,
    tight: bool = True,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()

    outputs: Dict[str, str] = {}
    for ext, kwargs in (
        ("png", {"format": "png", "dpi": DPI_PNG, "bbox_inches": "tight", "pad_inches": 0.02}),
        ("svg", {"format": "svg", "bbox_inches": "tight", "pad_inches": 0.02}),
    ):
        out = out_dir / f"{stem}.{ext}"
        fig.savefig(out, **kwargs)
        outputs[ext] = str(out.relative_to(REPO_ROOT))

    size = fig.get_size_inches()
    meta = {
        "figure": stem,
        "generating_script": str(script_path.relative_to(REPO_ROOT)),
        "git_commit": git_commit(),
        "source_files": [
            {"path": str(p.relative_to(REPO_ROOT)), "sha256": sha256_file(p)}
            for p in source_files
        ],
        "extracted_values": extracted_values,
        "output_width_pt": FIG_WIDTH_PT,
        "output_width_mm": FIG_WIDTH_MM,
        "output_height_mm": float(size[1] * 25.4),
        "output_dimensions_inches": {"width": float(size[0]), "height": float(size[1])},
        "dpi_png": DPI_PNG,
        "png_sha256": sha256_file(out_dir / f"{stem}.png"),
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": outputs,
    }
    if provenance:
        prov_path = out_dir / f"{stem}_provenance.json"
        prov_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    plt.close(fig)
    return meta


def format_sci(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 1e-3:
        return f"{value:.3g}"
    return f"{value:.2e}"
