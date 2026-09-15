"""Generate readability review assets from rendered figures and compiled PDF."""

from __future__ import annotations

import io
import re
import shutil
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from figure_style import (  # noqa: E402
    DPI_PNG,
    FIGURES_DIR,
    FIG_WIDTH_IN,
    READABILITY_REVIEW_DIR,
    REPO_ROOT,
    apply_style,
)


def _copy_manuscript_scale_pngs() -> list[Path]:
    READABILITY_REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for n in range(1, 6):
        src = FIGURES_DIR / f"Fig{n}.png"
        dst = READABILITY_REVIEW_DIR / f"Fig{n}_manuscript_scale.png"
        if not src.exists():
            raise FileNotFoundError(f"Missing rendered figure: {src}")
        shutil.copy2(src, dst)
        outputs.append(dst)
    return outputs


def _figure_pages_from_aux(aux_path: Path) -> list[int]:
    labels = [
        "fig:fali-overview",
        "fig:fali-game",
        "fig:validation",
        "fig:gpt2-attention-recruitment",
        "fig:cross-setting-reversal",
    ]
    text = aux_path.read_text()
    pages: list[int] = []
    for label in labels:
        match = re.search(rf"\\newlabel\{{{re.escape(label)}\}}\{{\{{(\d+)\}}\{{(\d+)\}}", text)
        if not match:
            raise RuntimeError(f"Missing aux label: {label}")
        pages.append(int(match.group(2)))
    return pages


def _figure_pages_from_pdf(pdf_path: Path) -> list[int]:
    aux = pdf_path.with_suffix(".aux")
    if aux.exists():
        return _figure_pages_from_aux(aux)
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.path.insert(0, str(REPO_ROOT / "tools/pydeps"))
        from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages: list[int] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if any(f"Fig. {n}" in text or f"Figure {n}" in text for n in range(1, 6)):
            if "List of Figures" in text:
                continue
            pages.append(idx)
    if len(pages) < 5:
        caption_pages = []
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if any(f"\\label{{fig:" in text for _ in [0]):
                pass
            if "fig:" in text.lower() and any(f"({chr(96+n)})" in text for n in range(1, 6)):
                caption_pages.append(idx)
        if len(caption_pages) >= 5:
            pages = caption_pages[:5]
    return pages


def _render_pdf_pages(pdf_path: Path, page_numbers: list[int], out_path: Path) -> Path:
    import fitz

    apply_style()
    doc = fitz.open(str(pdf_path))
    images = []
    for page in page_numbers:
        pix = doc.load_page(page - 1).get_pixmap(dpi=200)
        images.append(pix.tobytes("png"))

    fig, axes = plt.subplots(len(images), 1, figsize=(FIG_WIDTH_IN, FIG_WIDTH_IN * 0.55 * len(images)))
    if len(images) == 1:
        axes = [axes]
    for ax, png_bytes, page in zip(axes, images, page_numbers):
        arr = mpimg.imread(io.BytesIO(png_bytes), format="png")
        ax.imshow(arr)
        ax.axis("off")
        ax.set_title(f"main.pdf page {page}", fontsize=10, loc="left", pad=4)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.01, hspace=0.06)
    fig.savefig(out_path, dpi=DPI_PNG, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)
    doc.close()
    return out_path


def render(*, pdf_path: Path | None = None) -> tuple[list[Path], Path]:
    pngs = _copy_manuscript_scale_pngs()
    pdf = pdf_path or (REPO_ROOT / "manuscript/machine_learning/main.pdf")
    if not pdf.exists():
        raise FileNotFoundError(f"Compiled manuscript not found: {pdf}")
    pages = _figure_pages_from_pdf(pdf)
    if not pages:
        raise RuntimeError("Could not locate figure pages in main.pdf")
    contact = FIGURES_DIR / "compiled_figure_pages_contact_sheet.png"
    _render_pdf_pages(pdf, pages[:5], contact)
    return pngs, contact


if __name__ == "__main__":
    review_pngs, sheet = render()
    print("Manuscript-scale PNGs:")
    for p in review_pngs:
        print(p)
    print(f"Compiled-page contact sheet: {sheet}")
