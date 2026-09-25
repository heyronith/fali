#!/usr/bin/env python3
"""Build a flat, Overleaf-ready Machine Learning submission source tree.

Scientific source is the frozen Machine Learning conversion. This script makes only
journal/production changes: flat paths, SI metadata/Online Resource naming, state
fields required by the title-page guidance, and heading-depth normalization.
"""
from pathlib import Path
import re
import shutil

SOURCE_SCIENCE_HEAD = "7749367a17455a8a18725680f161cf63d3972dd4"
SRC = Path("manuscript/machine_learning")
OUT = Path("dist/FALI_Machine_Learning_Overleaf_Ready")

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

# Exact active manuscript source files.
for name in [
    "main.tex", "introduction.tex", "related_work.tex",
    "problem_formulation.tex", "fali_method.tex", "method_validation.tex",
    "gpt2_ioi.tex", "ablation_and_gauge.tex", "pythia_induction.tex",
    "discussion.tex", "limitations.tex", "conclusion.tex", "references.bib",
    "sn-jnl.cls", "sn-apacite.bst", "FALI_MLJ_Contribution_Information_Sheet.tex",
]:
    shutil.copy2(SRC / name, OUT / name)

# Rename the supplementary source to match the journal's Online Resource convention.
si_out = OUT / "ESM_1_Supplementary_Information.tex"
shutil.copy2(SRC / "FALI_Machine_Learning_Supplementary_Information.tex", si_out)

# Include publication PDF plus preferred vector EPS artwork in the flat package.
for n in range(1, 6):
    shutil.copy2(SRC / "figures" / f"fig{n}.tex", OUT / f"fig{n}.tex")
    shutil.copy2(SRC / "figures" / f"Fig{n}.pdf", OUT / f"Fig{n}.pdf")
    eps = SRC / "figures" / f"Fig{n}.eps"
    if eps.exists():
        shutil.copy2(eps, OUT / f"Fig{n}.eps")

# Flatten all LaTeX paths: Machine Learning explicitly says no subfolders.
for name in ["problem_formulation.tex", "fali_method.tex", "method_validation.tex", "gpt2_ioi.tex", "pythia_induction.tex"]:
    p = OUT / name
    s = p.read_text(encoding="utf-8")
    s = re.sub(r"\\input\{figures/(fig[1-5])\}", r"\\input{\1}", s)
    p.write_text(s, encoding="utf-8")
for n in range(1, 6):
    p = OUT / f"fig{n}.tex"
    s = p.read_text(encoding="utf-8").replace(f"figures/Fig{n}.pdf", f"Fig{n}.pdf")
    p.write_text(s, encoding="utf-8")

# Title-page geography: state is applicable for all three US affiliations.
p = OUT / "main.tex"
s = p.read_text(encoding="utf-8")
s = s.replace(r"\city{Weatherford}, \country{USA}", r"\city{Weatherford}, \state{Oklahoma}, \country{USA}")
s = s.replace(r"\city{Stillwater}, \country{USA}", r"\city{Stillwater}, \state{Oklahoma}, \country{USA}")

# Add a concise SI caption and explicit Online Resource citation.
resource_caption = (
    r"\section*{Supplementary Information}" + "\n"
    r"Online Resource 1: Supplementary Information containing additional details on Owen attribution, validation chronology, exact GPT-2 estimates, alternative-ablation and gauge-sensitivity analyses, and the Pythia-160M induction protocol." + "\n\n"
)
if "Online Resource 1:" not in s:
    marker = r"\section*{Acknowledgements}"
    if marker not in s:
        raise RuntimeError("Acknowledgements marker missing from main.tex")
    s = s.replace(marker, resource_caption + marker)
p.write_text(s, encoding="utf-8")

# Make in-text SI references specific, as requested by the journal.
replacements = {
    "method_validation.tex": [
        ("provided in the Supplementary Information", "provided in Online Resource 1"),
    ],
    "gpt2_ioi.tex": [
        ("given in the Supplementary Information", "given in Online Resource 1"),
    ],
    "ablation_and_gauge.tex": [
        ("reported in the Supplementary Information", "reported in Online Resource 1"),
        ("provided in the Supplementary Information", "provided in Online Resource 1"),
    ],
    "pythia_induction.tex": [
        ("described in the Supplementary Information", "described in Online Resource 1"),
    ],
}
for name, reps in replacements.items():
    p = OUT / name
    s = p.read_text(encoding="utf-8")
    for old, new in reps:
        s = s.replace(old, new)
    p.write_text(s, encoding="utf-8")

# The journal asks for no more than three heading levels. Current paragraph-style
# subheads are semantic subheads, so normalize them to level 2 rather than level 4.
p = OUT / "limitations.tex"
s = re.sub(r"\\paragraph\{([^}]+)\}\s*", r"\\subsection{\1}\n", p.read_text(encoding="utf-8"))
p.write_text(s, encoding="utf-8")
p = OUT / "method_validation.tex"
s = p.read_text(encoding="utf-8").replace(
    r"\paragraph{Use of generative AI tools.}", r"\subsection{Use of generative AI tools}"
)
p.write_text(s, encoding="utf-8")

# SI front matter: each supplementary file must identify article title, journal,
# authors, and the corresponding author's affiliation and email.
s = si_out.read_text(encoding="utf-8")
old = r'''\title{Supplementary Information for ``FALI: Mechanistic Decomposition of Conditional Attention-Head Causal Usefulness''}
\author*{Ronith Sharmila, Delen Dursun, Rupesh Agrawal}
\affil*{\orgname{Corresponding manuscript}, \orgaddress{\country{USA}}}'''
new = r'''\title{Supplementary Information for ``FALI: Mechanistic Decomposition of Conditional Attention-Head Causal Usefulness''}
\author*[1]{\fnm{Ronith} \sur{Sharmila}}\email{ronith.sharmila@swosu.edu}
\affil*[1]{\orgname{Southwestern Oklahoma State University}, \orgaddress{\city{Weatherford}, \state{Oklahoma}, \country{USA}}}
\author[2]{\fnm{Delen} \sur{Dursun}}
\affil[2]{\orgname{Oklahoma State University}, \orgaddress{\city{Stillwater}, \state{Oklahoma}, \country{USA}}}
\author[3]{\fnm{Rupesh} \sur{Agrawal}}
\affil[3]{\orgname{Northern Kentucky University}, \orgaddress{\city{Highland Heights}, \state{Kentucky}, \country{USA}}}'''
if old not in s:
    raise RuntimeError("Expected SI front matter not found; refusing silent rewrite")
s = s.replace(old, new)
s = s.replace(r"\maketitle" + "\n", r"\maketitle" + "\n" + r"\noindent\textbf{Journal:} Machine Learning\par" + "\n" + r"\medskip" + "\n", 1)
si_out.write_text(s, encoding="utf-8")

# Guardrails: package construction must not accidentally change scientific values.
required_tokens = {
    "gpt2_ioi.tex": ["0.382", "0.298", "0.279", "0.310", "-0.263", "0.783", "-0.341", "-0.471"],
    "pythia_induction.tex": ["14.017", "1.146", "0.621", "0.486", "-0.144", "+0.118", "0.514"],
    "ablation_and_gauge.tex": ["0.440", "-0.166", "0.265", "-0.0474", "-0.319", "0.644", "-0.202", "-0.047", "+0.021"],
}
for name, tokens in required_tokens.items():
    text = (OUT / name).read_text(encoding="utf-8")
    missing = [x for x in tokens if x not in text]
    if missing:
        raise RuntimeError(f"Scientific guardrail failed for {name}: {missing}")

readme = f'''FALI — Machine Learning (Springer) — Overleaf-ready source package

Scientific manuscript source frozen at: {SOURCE_SCIENCE_HEAD}
Package construction: production/journal compliance only; no new experiments.

UPLOAD TO OVERLEAF
1. New Project -> Upload Project.
2. Upload FALI_Machine_Learning_Overleaf_Ready.zip.
3. Compiler: pdfLaTeX.
4. Main document: main.tex.
5. Compile.

OTHER DOCUMENTS IN THE PROJECT
- ESM_1_Supplementary_Information.tex (compile/download as ESM_1_Supplementary_Information.pdf for journal SI)
- FALI_MLJ_Contribution_Information_Sheet.tex

PACKAGE DESIGN
- flat directory: no LaTeX subfolders
- Springer Nature sn-jnl.cls and sn-apacite.bst included
- references.bib included
- five main figures included as vector PDF; EPS production artwork also included
- captions are in LaTeX, outside the figure artwork
- SI contains article title, journal, authors, corresponding-author affiliation and email
- main text explicitly cites SI as Online Resource 1
- headings normalized to <=3 levels

AUTHOR ITEMS TO COMPLETE BEFORE ACTUAL JOURNAL SUBMISSION IF AVAILABLE
- ORCID IDs (the journal requests them if available)
- replace the future-tense Data and Code Availability wording with the final public repository/archive URL or persistent DOI once deposited

After Overleaf compilation, return the exact main PDF for page-by-page visual approval.
'''
(OUT / "README.txt").write_text(readme, encoding="utf-8")

print(f"Built {OUT}")
