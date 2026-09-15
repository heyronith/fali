"""Create review contact sheets for Figures 1–5."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from figure_style import DPI_PNG, FIG_WIDTH_IN, FIGURES_DIR, apply_style  # noqa: E402


def _render_sheet(pngs, out_path: Path, *, fig_width: float, dpi: int) -> Path:
    apply_style()
    fig, axes = plt.subplots(5, 1, figsize=(fig_width, fig_width * 2.2))
    for ax, png in zip(axes, pngs):
        img = mpimg.imread(png)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(png.stem, fontsize=10, loc="left", pad=6)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, hspace=0.08)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    plt.close(fig)
    return out_path


def render() -> tuple[Path, Path]:
    pngs = [FIGURES_DIR / f"Fig{i}.png" for i in range(1, 6)]
    missing = [p for p in pngs if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing PNG previews:\n" + "\n".join(str(p) for p in missing))

    review = _render_sheet(pngs, FIGURES_DIR / "figures_contact_sheet.png", fig_width=8.5, dpi=250)
    printsize = _render_sheet(
        pngs,
        FIGURES_DIR / "figures_contact_sheet_printsize.png",
        fig_width=FIG_WIDTH_IN,
        dpi=DPI_PNG,
    )
    return review, printsize


if __name__ == "__main__":
    review, printsize = render()
    print(review)
    print(printsize)
