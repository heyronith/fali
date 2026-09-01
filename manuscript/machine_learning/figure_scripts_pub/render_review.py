"""Contact sheet and manuscript-page review renders for publication figures."""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

SCRIPTS = Path(__file__).resolve().parent
ML = SCRIPTS.parent
REPO = ML.parents[1]
FIGURES = ML / "figures"
CANDIDATE = FIGURES / "candidate_final"
REVIEW = FIGURES / "publication_review"

_TW = json.loads((SCRIPTS / "textwidth.json").read_text())
FIG_WIDTH_IN = float(_TW["textwidth_in"])
DPI = int(_TW["dpi_png"])


def _figure_pages(aux: Path) -> list[int]:
    text = aux.read_text()
    labels = [
        "fig:fali-overview",
        "fig:fali-game",
        "fig:validation",
        "fig:gpt2-attention-recruitment",
        "fig:cross-setting-reversal",
    ]
    pages = []
    for label in labels:
        m = re.search(rf"\\newlabel\{{{re.escape(label)}\}}\{{\{{(\d+)\}}\{{(\d+)\}}", text)
        if not m:
            raise RuntimeError(f"missing label {label}")
        pages.append(int(m.group(2)))
    return pages


def contact_sheet(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pngs = [FIGURES / f"Fig{n}.png" for n in range(1, 6)]
    fig, axes = plt.subplots(5, 1, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 2.4))
    for ax, png in zip(axes, pngs):
        ax.imshow(mpimg.imread(png))
        ax.axis("off")
        ax.set_title(png.stem, fontsize=9, loc="left")
    fig.subplots_adjust(hspace=0.06)
    out = out_dir / "figures_contact_sheet_printsize.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def manuscript_pages(pdf: Path, out_dir: Path) -> tuple[Path, list[Path]]:
    import fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    pages = _figure_pages(pdf.with_suffix(".aux"))
    doc = fitz.open(str(pdf))
    singles = []
    for i, page_num in enumerate(pages, start=1):
        pix = doc.load_page(page_num - 1).get_pixmap(dpi=200)
        out = out_dir / f"Fig{i}_manuscript_page.png"
        out.write_bytes(pix.tobytes("png"))
        singles.append(out)
    images = [mpimg.imread(io.BytesIO(doc.load_page(p - 1).get_pixmap(dpi=150).tobytes("png"))) for p in pages]
    fig, axes = plt.subplots(len(images), 1, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.55 * len(images)))
    if len(images) == 1:
        axes = [axes]
    for ax, img, p in zip(axes, images, pages):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"main.pdf page {p}", fontsize=9, loc="left")
    fig.subplots_adjust(hspace=0.05)
    sheet = out_dir / "compiled_figure_pages_contact_sheet.png"
    fig.savefig(sheet, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    doc.close()
    return sheet, singles


if __name__ == "__main__":
    REVIEW.mkdir(parents=True, exist_ok=True)
    cs = contact_sheet(REVIEW)
    sheet, singles = manuscript_pages(ML / "main.pdf", REVIEW)
    print(cs)
    print(sheet)
    for p in singles:
        print(p)
