#!/usr/bin/env python3
"""Deterministically build a flat Overleaf-ready MLJ LaTeX source package.

Scientific/source baseline is frozen at SOURCE_HEAD. Only submission-formatting
changes are made on copied files; scientific claims, estimates, citations, and
figure contents are not changed.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

SOURCE_HEAD = "7749367a17455a8a18725680f161cf63d3972dd4"
PACKAGE_NAME = "FALI_Machine_Learning_Overleaf_Ready"
HERE = Path(__file__).resolve().parent

SECTIONS = [
    "introduction.tex", "related_work.tex", "problem_formulation.tex",
    "fali_method.tex", "method_validation.tex", "gpt2_ioi.tex",
    "ablation_and_gauge.tex", "pythia_induction.tex", "discussion.tex",
    "limitations.tex", "conclusion.tex",
]
ROOT_FILES = [
    "main.tex", *SECTIONS, "references.bib", "sn-jnl.cls",
    "sn-apacite.bst", "FALI_MLJ_Contribution_Information_Sheet.tex",
]
SI_SOURCE = "FALI_Machine_Learning_Supplementary_Information.tex"
SI_OUTPUT = "ESM_1_Supplementary_Information.tex"
CIS_OUTPUT = "FALI_MLJ_Contribution_Information_Sheet.tex"
FIGS = [f"Fig{i}" for i in range(1, 6)]


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8", newline="\n")


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return "UNKNOWN"


def transform_section(text: str) -> str:
    text = re.sub(r"\\input\{figures/(fig[1-5])\}", r"\\input{\1}", text)
    text = text.replace(
        "the Supplementary Information",
        "Online Resource~1 (Supplementary Information)",
    )
    return text


def transform_wrapper(text: str) -> str:
    return re.sub(r"\{figures/(Fig[1-5]\.pdf)\}", r"{\1}", text)


def transform_si(text: str) -> str:
    old = r"""\author*{Ronith Sharmila, Delen Dursun, Rupesh Agrawal}
\affil*{\orgname{Corresponding manuscript}, \orgaddress{\country{USA}}}
"""
    new = r"""\author*[1]{\fnm{Ronith} \sur{Sharmila}}\email{ronith.sharmila@swosu.edu}
\affil*[1]{\orgname{Southwestern Oklahoma State University}, \orgaddress{\city{Weatherford}, \country{USA}}}

\author[2]{\fnm{Delen} \sur{Dursun}}\email{dursun.delen@okstate.edu}
\affil[2]{\orgname{Oklahoma State University}, \orgaddress{\city{Stillwater}, \country{USA}}}

\author[3]{\fnm{Rupesh} \sur{Agrawal}}\email{rupesh.agrawal@nku.edu}
\affil[3]{\orgname{Northern Kentucky University}, \orgaddress{\city{Highland Heights}, \state{Kentucky}, \country{USA}}}
"""
    if old not in text:
        raise RuntimeError("SI author/affiliation block not found")
    text = text.replace(old, new, 1)
    marker = "\\maketitle\n"
    if marker not in text:
        raise RuntimeError("SI \\maketitle not found")
    text = text.replace(
        marker,
        marker
        + "\\noindent\\textbf{Journal:} Machine Learning (Springer Nature)\\\\\n"
        + "\\textbf{Online Resource:} 1\\par\n\\medskip\n",
        1,
    )
    # sn-jnl already loads/configures geometry.
    text = text.replace("\\usepackage{geometry}\n", "")
    return text


def make_readme(pkg: Path) -> None:
    write(pkg / "README_OVERLEAF.txt", f"""FALI — MACHINE LEARNING (SPRINGER NATURE) — OVERLEAF-READY SOURCE

Scientific/source baseline commit:
{SOURCE_HEAD}

OVERLEAF
1. New Project -> Upload Project -> upload this ZIP.
2. Compiler: pdfLaTeX.
3. Main document: main.tex.
4. Recompile; Overleaf/latexmk should run BibTeX automatically.
5. Download the exact compiled main PDF and return it for final page-by-page visual QA.

SEPARATE DOCUMENTS
- Online Resource 1 / SI: {SI_OUTPUT}
- MLJ Contribution Information Sheet: {CIS_OUTPUT}

PACKAGE RULES
- Intentionally flat: no subdirectories.
- Fig1-Fig5 are copied unchanged from the source baseline.
- PDFs are used for pdflatex; matching EPS vector assets are also supplied.
- Scientific claims, numbers, citations, and experiments are unchanged.
- Only SI metadata, Online Resource wording, and path flattening are normalized.
- ORCID IDs, postal addresses, and persistent data/code identifiers are not invented.

Do not submit automatically. Compile and visually inspect first.
""")


def make_audit(pkg: Path) -> None:
    write(pkg / "MLJ_REQUIREMENTS_AUDIT.md", f"""# Independent Machine Learning submission-format audit

**Scientific/source baseline:** `{SOURCE_HEAD}`  
**Packaging branch HEAD at build:** `{git_head()}`

## Package-level checks
- MLJ Contribution Information Sheet included; final prior-overlap statement required.
- `sn-jnl` + `pdflatex` + `sn-apa` retained.
- Abstract validator enforces 150–250 words.
- Keyword validator enforces 4–6 keywords.
- Corresponding author email and affiliations retained.
- Statements and Declarations / Competing Interests retained.
- Substantive LLM use disclosure retained in Methods material.
- Source ZIP is flat; no figure/bibliography subfolders.
- `references.bib` and `sn-apacite.bst` included.
- Fig1–Fig5 included as PDF compilation assets and EPS vector assets.
- SI normalized as Online Resource 1 with article/journal/authors/affiliations/corresponding email.
- Main-text SI mentions normalized to Online Resource 1 wording.
- Active main source is checked for stale Elsevier/Neural Networks venue strings.

## Deliberately not fabricated
- ORCID IDs.
- Postal street addresses.
- Persistent data/code DOI or URL.

## Final gate
Compile `main.tex` in Overleaf and inspect the exact resulting PDF page by page before submission.
""")


def make_provenance(pkg: Path) -> None:
    lines = [
        "FALI Machine Learning Overleaf package provenance",
        f"Scientific/source baseline: {SOURCE_HEAD}",
        f"Packaging branch HEAD at build: {git_head()}",
        "",
        "Research-lead PDF figure SHA256 fingerprints:",
    ]
    for stem in FIGS:
        p = HERE / "figures" / f"{stem}.pdf"
        lines.append(f"{stem}.pdf  {hashlib.sha256(p.read_bytes()).hexdigest()}")
    write(pkg / "SOURCE_PROVENANCE.txt", "\n".join(lines) + "\n")


def validate(pkg: Path) -> None:
    failures: list[str] = []
    if any(p.is_file() and p.parent != pkg for p in pkg.rglob("*")):
        failures.append("package is not flat")

    required = {
        "main.tex", *SECTIONS, *[f"fig{i}.tex" for i in range(1, 6)],
        *[f"Fig{i}.pdf" for i in range(1, 6)], *[f"Fig{i}.eps" for i in range(1, 6)],
        "references.bib", "sn-jnl.cls", "sn-apacite.bst", SI_OUTPUT, CIS_OUTPUT,
        "README_OVERLEAF.txt", "MLJ_REQUIREMENTS_AUDIT.md", "SOURCE_PROVENANCE.txt",
    }
    missing = sorted(x for x in required if not (pkg / x).is_file())
    if missing:
        failures.append("missing: " + ", ".join(missing))

    main = read(pkg / "main.tex")
    section_text = "\n".join(read(pkg / x) for x in SECTIONS)
    active = main + "\n" + section_text

    if "\\documentclass[pdflatex,sn-apa]{sn-jnl}" not in main:
        failures.append("wrong documentclass/bibliography mode")
    if "\\bibliography{references}" not in main:
        failures.append("references.bib not referenced")
    if "Statements and Declarations" not in main or "Competing Interests" not in main:
        failures.append("required declarations missing")
    if "Use of generative AI tools" not in active:
        failures.append("LLM disclosure missing")
    if "figures/" in active:
        failures.append("main source still contains figures/ path")
    if "Neural Networks" in active or "Elsevier" in active:
        failures.append("stale prior-venue string in active main source")

    am = re.search(r"\\abstract\{(.*?)\}\s*\\keywords", main, re.S)
    if not am:
        failures.append("abstract parse failed")
    else:
        words = re.findall(r"\b[A-Za-z0-9][A-Za-z0-9'-]*\b", am.group(1))
        if not 150 <= len(words) <= 250:
            failures.append(f"abstract words={len(words)}")

    km = re.search(r"\\keywords\{(.*?)\}", main, re.S)
    if not km:
        failures.append("keyword parse failed")
    else:
        n = len([x for x in km.group(1).split(",") if x.strip()])
        if not 4 <= n <= 6:
            failures.append(f"keywords={n}")

    for i in range(1, 6):
        w = read(pkg / f"fig{i}.tex")
        if f"{{Fig{i}.pdf}}" not in w or "figures/" in w:
            failures.append(f"fig{i}.tex path invalid")

    si = read(pkg / SI_OUTPUT)
    for token in [
        "Machine Learning (Springer Nature)", "Online Resource:} 1",
        "ronith.sharmila@swosu.edu", "Southwestern Oklahoma State University",
        "Oklahoma State University", "Northern Kentucky University",
    ]:
        if token not in si:
            failures.append("SI metadata missing: " + token)

    cis = read(pkg / CIS_OUTPUT)
    if "Author confirmation before submission" in cis:
        failures.append("CIS placeholder remains")
    if "No additional conference, workshop, journal, or archival version overlaps" not in cis:
        failures.append("CIS final overlap statement missing")

    if failures:
        raise RuntimeError("Package validation failed:\n" + "\n".join("- " + x for x in failures))


def make_sums(pkg: Path) -> None:
    lines = []
    for p in sorted(pkg.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    write(pkg / "SHA256SUMS.txt", "\n".join(lines) + "\n")


def build(out: Path) -> Path:
    pkg = out / PACKAGE_NAME
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)

    for name in ROOT_FILES:
        src = HERE / name
        if not src.is_file():
            raise FileNotFoundError(src)
        text = read(src)
        if name in SECTIONS:
            text = transform_section(text)
        if name == "main.tex":
            text = text.replace("\\usepackage{geometry}\n", "")
        write(pkg / name, text)

    for i, stem in enumerate(FIGS, 1):
        write(pkg / f"fig{i}.tex", transform_wrapper(read(HERE / "figures" / f"fig{i}.tex")))
        for ext in ("pdf", "eps"):
            shutil.copy2(HERE / "figures" / f"{stem}.{ext}", pkg / f"{stem}.{ext}")

    write(pkg / SI_OUTPUT, transform_si(read(HERE / SI_SOURCE)))
    make_readme(pkg)
    make_audit(pkg)
    make_provenance(pkg)
    validate(pkg)
    make_sums(pkg)

    zpath = out / f"{PACKAGE_NAME}.zip"
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(pkg.iterdir(), key=lambda x: x.name.lower()):
            if p.is_file():
                z.write(p, p.name)
    with zipfile.ZipFile(zpath) as z:
        if any("/" in n.rstrip("/") for n in z.namelist()):
            raise RuntimeError("ZIP is not flat")
    print(zpath)
    return zpath


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "_overleaf_artifacts")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    build(args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
