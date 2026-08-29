"""Create a review contact sheet for Figures 1–5."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from figure_style import FIGURES_DIR, apply_style  # noqa: E402


def render() -> Path:
    apply_style()
    pngs = [FIGURES_DIR / f"Fig{i}.png" for i in range(1, 6)]
    missing = [p for p in pngs if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing PNG previews:\n" + "\n".join(str(p) for p in missing))

    fig, axes = plt.subplots(5, 1, figsize=(8.5, 14))
    for ax, png in zip(axes, pngs):
        img = mpimg.imread(png)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(png.stem, fontsize=10, loc="left")
    fig.tight_layout()
    out = FIGURES_DIR / "figures_contact_sheet.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    path = render()
    print(path)
