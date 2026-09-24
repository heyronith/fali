#!/usr/bin/env python3
"""Build a flat, Overleaf-ready Machine Learning (Springer Nature) LaTeX package.

Scientific source is frozen at SOURCE_HEAD. This builder performs only deterministic
submission-formatting operations on copies of the manuscript sources:
- flattens figure wrapper/source paths (Springer submission requirement),
- adds explicit Online Resource 1 wording in main-text SI references,
- gives the SI complete article/journal/author/affiliation/corresponding-email metadata,
- renames the SI source to an ESM-style filename,
- copies the exact research-lead figure assets already committed at SOURCE_HEAD,
- emits provenance, audit, and SHA-256 manifests.

It does not alter scientific estimates, claims, citations, or figure contents.
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

MAIN_SECTION_FILES = [
    "introduction.tex",
    "related_work.tex",
    "problem_formulation.tex",
    "fali_method.tex",
    "method_validation.tex",
    "gpt2_ioi.tex",
    "ablation_and_gauge.tex",
    "pythia_induction.tex",
    "discussion.tex",
    "limitations.tex",
    "conclusion.tex",
]

ROOT_TEXT_FILES = [
    "main.tex",
    *MAIN_SECTION_FILES,
    "references.bib",
    "sn-jnl.cls",
    "sn-apacite.bst",
    "FALI_MLJ_Contribution_Information_Sheet.tex",
]

FIGURE_STEMS = ["Fig1", "Fig2", "Fig3", "Fig4", "Fig5"]

SI_SOURCE = "FALI_Machine_Learning_Supplementary_Information.tex"
SI_OUTPUT = "ESM_1_Supplementary_Information.tex"
CIS_OUTPUT = "FALI_MLJ_Contribution_Information_Sheet.tex"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def source_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def transform_main_section(text: str) -> str:
    # Submission ZIP must be flat: figure wrappers are copied to ZIP root.
    text = re.sub(r"\\input\{figures/(fig[1-5])\}", r"\\input{\1}", text)

    # Springer SI instructions ask that the supplement be cited as an Online Resource.
    # Apply only to main-manuscript section files, never to the SI itself.
    text = text.replace(
        "the Supplementary Information",
        "Online Resource~1 (Supplementary Information)",
    )
    return text


def transform_figure_wrapper(text: str) -> str:
    # PDF remains the pdflatex compilation asset. EPS is included separately as
    # the preferred publication vector format but is not referenced by pdflatex.
    text = re.sub(r"\{figures/(Fig[1-5]\.pdf)\}", r"{\1}", text)
    return text


def transform_si(text: str) -> str:
    # Keep all scientific SI content untouched. Replace only front matter to meet
    # Machine Learning's SI metadata instructions.
    pattern = re.compile(
        r"\\author\*\{Ronith Sharmila, Delen Dursun, Rupesh Agrawal\}\s*"
        r"\\affil\*\{\\orgname\{Corresponding manuscript\}, "
        r"\\orgaddress\{\\country\{USA\}\}\}\s*",
        flags=re.MULTILINE,
    )
    replacement = r"""\author*[1]{\fnm{Ronith} \sur{Sharmila}}\email{ronith.sharmila@swosu.edu}
\affil*[1]{\orgname{Southwestern Oklahoma State University}, \orgaddress{\city{Weatherford}, \country{USA}}}

\author[2]{\fnm{Delen} \sur{Dursun}}\email{dursun.delen@okstate.edu}
\affil[2]{\orgname{Oklahoma State University}, \orgaddress{\city{Stillwater}, \country{USA}}}

\author[3]{\fnm{Rupesh} \sur{Agrawal}}\email{rupesh.agrawal@nku.edu}
\affil[3]{\orgname{Northern Kentucky University}, \orgaddress{\city{Highland Heights}, \state{Kentucky}, \country{USA}}}
"""
    text, n = pattern.subn(replacement, text, count=1)
    if n != 1:
        raise RuntimeError("Could not locate SI author/affiliation front matter")

    marker = "\\maketitle\n"
    if marker not in text:
        raise RuntimeError("SI is missing \\maketitle")
    text = text.replace(
        marker,
        marker
        + "\\noindent\\textbf{Journal:} Machine Learning (Springer Nature)\\\\\n"
        + "\\textbf{Online Resource:} 1\\par\n\\medskip\n",
        1,
    )

    # The class already loads/configures geometry. Avoid a redundant second load.
    text = text.replace("\\usepackage{geometry}\n", "")
    return text


def validate_package(pkg: Path) -> None:
    failures: list[str] = []

    # Flatness: Springer instructs LaTeX submissions not to use subfolders.
    nested = [p for p in pkg.rglob("*") if p.is_file() and p.parent != pkg]
    if nested:
        failures.append("Package contains nested files: " + ", ".join(map(str, nested)))

    required = {
        "main.tex",
        *MAIN_SECTION_FILES,
        *[f"fig{i}.tex" for i in range(1, 6)],
        *[f"Fig{i}.pdf" for i in range(1, 6)],
        *[f"Fig{i}.eps" for i in range(1, 6)],
        "references.bib",
        "sn-jnl.cls",
        "sn-apacite.bst",
        SI_OUTPUT,
        CIS_OUTPUT,
        "README_OVERLEAF.txt",
        "MLJ_REQUIREMENTS_AUDIT.md",
        "SOURCE_PROVENANCE.txt",
    }
    missing = sorted(name for name in required if not (pkg / name).exists())
    if missing:
        failures.append("Missing required files: " + ", ".join(missing))

    main = read_text(pkg / "main.tex")
    joined_main = main + "\n" + "\n".join(read_text(pkg / f) for f in MAIN_SECTION_FILES)

    if "\\documentclass[pdflatex,sn-apa]{sn-jnl}" not in main:
        failures.append("main.tex is not configured for sn-jnl + sn-apa + pdflatex")
    if "\\bibliography{references}" not in main:
        failures.append("main.tex does not reference references.bib")
    if "Statements and Declarations" not in main or "Competing Interests" not in main:
        failures.append("Required Statements and Declarations / Competing Interests missing")
    if "Use of generative AI tools" not in joined_main:
        failures.append("LLM-use disclosure not found in manuscript methods")
    if "figures/" in joined_main:
        failures.append("A main-manuscript source still contains a figures/ subfolder path")

    # Abstract word count.
    m = re.search(r"\\abstract\{(.*?)\}\s*\\keywords", main, flags=re.DOTALL)
    if not m:
        failures.append("Could not parse abstract")
    else:
        abstract = re.sub(r"\\[A-Za-z]+(?:\{[^{}]*\})?", " ", m.group(1))
        words = re.findall(r"\b[\w'-]+\b", abstract)
        if not (150 <= len(words) <= 250):
            failures.append(f"Abstract word count outside 150-250: {len(words)}")

    km = re.search(r"\\keywords\{(.*?)\}", main, flags=re.DOTALL)
    if not km:
        failures.append("Could not parse keywords")
    else:
        k = [x.strip() for x in km.group(1).split(",") if x.strip()]
        if not (4 <= len(k) <= 6):
            failures.append(f"Keyword count outside 4-6: {len(k)}")

    # Figure wrappers must point to local files and all five figures must be present.
    for i in range(1, 6):
        wrapper = read_text(pkg / f"fig{i}.tex")
        if f"{{Fig{i}.pdf}}" not in wrapper:
            failures.append(f"fig{i}.tex does not reference local Fig{i}.pdf")
        if "figures/" in wrapper:
            failures.append(f"fig{i}.tex retains a subfolder path")

    si = read_text(pkg / SI_OUTPUT)
    for token in [
        "Machine Learning (Springer Nature)",
        "Online Resource:} 1",
        "ronith.sharmila@swosu.edu",
        "Southwestern Oklahoma State University",
        "Oklahoma State University",
        "Northern Kentucky University",
    ]:
        if token not in si:
            failures.append(f"SI metadata missing: {token}")

    cis = read_text(pkg / CIS_OUTPUT)
    if "Author confirmation before submission" in cis:
        failures.append("CIS still contains author-confirmation placeholder")
    if "No additional conference, workshop, journal, or archival version overlaps" not in cis:
        failures.append("CIS final overlap statement missing")

    if "Neural Networks" in joined_main or "Elsevier" in joined_main:
        failures.append("Stale Elsevier/Neural Networks venue string in active main manuscript")

    if failures:
        raise RuntimeError("\n".join(f"- {x}" for x in failures))


def write_readme(pkg: Path) -> None:
    text = f"""FALI — MACHINE LEARNING (SPRINGER NATURE) — OVERLEAF-READY SOURCE

Scientific/source baseline commit:
{SOURCE_HEAD}

Target journal:
Machine Learning (Springer Nature)

OVERLEAF
1. Create a new project using "Upload Project" and upload the ZIP containing these files.
2. Set Compiler = pdfLaTeX.
3. Set Main document = main.tex.
4. Recompile. Overleaf/latexmk should run BibTeX automatically for references.bib.
5. Download the compiled main PDF and return that exact PDF for final page-by-page visual QA.

SEPARATE COMPILATIONS
- Supplementary Information: set Main document to {SI_OUTPUT} and compile with pdfLaTeX.
- MLJ Contribution Information Sheet: set Main document to {CIS_OUTPUT} and compile with pdfLaTeX.

SUBMISSION-SAFETY NOTES
- This ZIP is intentionally FLAT: no subdirectories.
- Fig1–Fig5 are the exact research-lead assets from the source baseline; figure contents are not regenerated here.
- PDF figure files are used for pdflatex compilation; matching EPS files are included as publication vector assets.
- Scientific claims, numerical results, citations, and experimental content are unchanged.
- SI front matter and main-text SI references receive only venue-compliance wording/metadata changes.
- ORCID identifiers may be added if available; the journal lists ORCID as optional.
- Replace/update the Data and Code Availability statement only if a persistent repository/DOI is available at submission time.

DO NOT submit automatically. First compile in Overleaf and visually inspect the resulting PDF.
"""
    write_text(pkg / "README_OVERLEAF.txt", text)


def write_audit(pkg: Path) -> None:
    audit = f"""# Independent Machine Learning submission-format audit

**Scientific/source baseline:** `{SOURCE_HEAD}`  
**Package builder commit at run time:** `{source_commit()}`  
**Target:** *Machine Learning* (Springer Nature)

## PASS — packaged requirements

- Mandatory MLJ Contribution Information Sheet included; final prior-overlap statement present.
- Springer Nature `sn-jnl` class retained with `pdflatex` and `sn-apa` name-year bibliography configuration.
- Abstract is validated to the journal's 150–250 word range.
- Keyword count is validated to 4–6.
- Corresponding author email and author affiliations are present.
- Statements and Declarations / Competing Interests are present.
- Substantive LLM use is disclosed in the Methods material.
- Package is flat: no LaTeX subfolders or subfolder figure paths.
- References use the existing `references.bib` plus `sn-apacite.bst`.
- Five main figures are included as local PDF compilation assets and EPS publication assets.
- SI is explicitly identified as Online Resource 1 and includes article/journal/authors/affiliations/corresponding email.
- Main-text references to the Supplementary Information are normalized to Online Resource 1 wording.
- CIS placeholder language is absent.
- No active main-manuscript Elsevier/Neural Networks venue strings are permitted by the package validator.

## Deliberately not fabricated

- ORCID IDs are not added unless supplied by the authors.
- A persistent data/code DOI or URL is not invented; the existing availability statement is preserved.
- Postal street addresses are not invented.

## Final gate still required

Compile `main.tex` in Overleaf, download the exact resulting PDF, and perform a page-by-page visual inspection before submission. Compilation logs alone do not establish visual correctness.
"""
    write_text(pkg / "MLJ_REQUIREMENTS_AUDIT.md", audit)


def write_provenance(pkg: Path) -> None:
    fig_lines = []
    for stem in FIGURE_STEMS:
        p = HERE / "figures" / f"{stem}.pdf"
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        fig_lines.append(f"{stem}.pdf SHA256 {digest}")
    text = (
        "FALI Machine Learning Overleaf package provenance\n"
        f"Scientific/source baseline: {SOURCE_HEAD}\n"
        f"Packaging branch HEAD at build: {source_commit()}\n"
        "\nResearch-lead figure fingerprints (source repository files):\n"
        + "\n".join(fig_lines)
        + "\n"
    )
    write_text(pkg / "SOURCE_PROVENANCE.txt", text)


def write_sha256s(pkg: Path) -> None:
    lines = []
    for p in sorted(pkg.iterdir(), key=lambda x: x.name.lower()):
        if not p.is_file() or p.name == "SHA256SUMS.txt":
            continue
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    write_text(pkg / "SHA256SUMS.txt", "\n".join(lines) + "\n")


def build(out_dir: Path) -> Path:
    pkg = out_dir / PACKAGE_NAME
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True, exist_ok=True)

    # Root manuscript text/style assets.
    for name in ROOT_TEXT_FILES:
        src = HERE / name
        if not src.exists():
            raise FileNotFoundError(src)
        text = read_text(src)
        if name in MAIN_SECTION_FILES:
            text = transform_main_section(text)
        if name == "main.tex":
            # sn-jnl already loads/configures geometry internally.
            text = text.replace("\\usepackage{geometry}\n", "")
        write_text(pkg / name, text)

    # Flatten figure wrappers + exact current figure assets.
    for i, stem in enumerate(FIGURE_STEMS, start=1):
        wrapper = read_text(HERE / "figures" / f"fig{i}.tex")
        write_text(pkg / f"fig{i}.tex", transform_figure_wrapper(wrapper))
        for ext in ("pdf", "eps"):
            src = HERE / "figures" / f"{stem}.{ext}"
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, pkg / src.name)

    # SI: scientific body unchanged; submission-facing metadata normalized.
    si = transform_si(read_text(HERE / SI_SOURCE))
    write_text(pkg / SI_OUTPUT, si)

    write_readme(pkg)
    write_audit(pkg)
    write_provenance(pkg)

    validate_package(pkg)
    write_sha256s(pkg)

    zip_path = out_dir / f"{PACKAGE_NAME}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(pkg.iterdir(), key=lambda x: x.name.lower()):
            if p.is_file():
                # ZIP root is intentionally flat.
                zf.write(p, arcname=p.name)

    # Verify ZIP itself is flat and complete.
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if any("/" in n.rstrip("/") for n in names):
            raise RuntimeError("ZIP is not flat")
        if "main.tex" not in names or SI_OUTPUT not in names or CIS_OUTPUT not in names:
            raise RuntimeError("ZIP missing core documents")

    print(zip_path)
    return zip_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=HERE / "_overleaf_artifacts",
        help="Directory for generated package directory and ZIP",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    build(args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
