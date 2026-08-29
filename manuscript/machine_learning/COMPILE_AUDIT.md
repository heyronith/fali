# Manuscript Compile Audit

**Branch:** `manuscript/reconstruction`  
**Integration commit base:** `1889ce34705e5c9c5d64a6c6a74ea0b6ccc68793`  
**Status:** **PASS**

## Official Springer Nature template provenance

Downloaded from Springer Nature LaTeX Author Support, **journal article template package (December 2024 version)**:

- **Page:** https://www.springernature.com/gp/authors/campaigns/latex-author-support/see-where-our-services-will-take-you/18782940
- **CMS URL:** `https://resource-cms.springernature.com/springer-cms/rest/v1/content/18782940/data/v12`
- **Archive filename:** `sn-journal-template.zip`
- **Archive SHA256:** `812e76dcaa9c28dc1bff1fb6065d51729b67d4ea140552a05088317414a3ecae`
- **`sn-jnl.cls` SHA256:** `36d0c3273a59d48dc6a9c7b080dfa1ec50dc10229d8751568d1f2e490ffa5ecc`
- **`sn-mathphys-ay.bst` SHA256:** `6a2306683997db76b5e58190acea4dac360c742044114f6a923a4380fbab9008`

Unmodified copies placed in `manuscript/machine_learning/` per `README.md`. LPPL-licensed Springer Nature template assets are tracked in Git for reproducible local compilation.

## LaTeX toolchain

Installed via user-local TinyTeX at `/Users/ronny/fali-tinytex` (no sudo required on macOS).

| Tool | Version |
|------|---------|
| `pdflatex` | pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026) |
| `latexmk` | John Collins, 9 March 2026. Version 4.88 |
| `bibtex` | BibTeX 0.99e (TeX Live 2026) |

## Compile commands

From `manuscript/machine_learning/`:

```bash
export PATH="/Users/ronny/fali-tinytex/bin/universal-darwin:$PATH"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error contribution_information_sheet.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

## Compile results

| Document | Status | Output | Pages |
|----------|--------|--------|-------|
| `main.tex` | **PASS** | `main.pdf` | 24 |
| `contribution_information_sheet.tex` | **PASS** | `contribution_information_sheet.pdf` | 2 |
| `supplement.tex` | **PLACEHOLDER-BUT-COMPILES** | `supplement.pdf` | 1 |

## LaTeX plumbing fixes (non-scientific)

| Fix | File | Reason |
|-----|------|--------|
| Removed `\jyear{2026}` | `main.tex` | Undefined in official December-2024 `sn-jnl.cls`; not part of Springer template API |

No Springer class/style files were edited. No scientific prose, results, or figure data were changed.

## Bibliography

| Check | Result |
|-------|--------|
| `references.bib` found | yes |
| Style `sn-mathphys-ay.bst` used | yes |
| BibTeX completed | yes |
| Author--year citations render | yes |
| Bibliography in PDF | yes (pages 22--24) |
| `?` / `??` citations | none observed |

## Page budget

| Metric | Value |
|--------|-------|
| Total `main.pdf` pages | 24 |
| References begin | page 22 |
| Bibliography page count | 3 (pages 22--24) |
| Approx. scientific main text (title through conclusion) | ~21 pages |

**Classification:** **PAGE-BUDGET CONCERN** (23--24 band; internal target ~20 including references)

Space is consumed primarily by: long FALI formalism (Sec. 4), five full-width figures, and a 3-page alphabetized reference list (20 entries).

## Figure pages and float placement

| Figure | Label | First textual mention | Figure page | Page distance | Notes |
|--------|-------|----------------------|-------------|---------------|-------|
| Fig 1 | `fig:fali-overview` | Sec. 3.2 (~p. 7) | 8 | 1 | OK |
| Fig 2 | `fig:fali-game` | Sec. 4.2 (~p. 9) | 10 | 1 | OK |
| Fig 3 | `fig:validation` | Sec. 5 opening (~p. 11) | 12 | 1 | OK |
| Fig 4 | `fig:gpt2-attention-recruitment` | Sec. 6.3 (~p. 15); no `\ref` in `gpt2_ioi.tex` | 16 | ~1 | Missing explicit in-text `\ref`; figure follows subsection |
| Fig 5 | `fig:cross-setting-reversal` | Sec. 7.2 (~p. 17) | 19 | 2 | OK; float drifts into Discussion (Sec. 8 starts p. 18) |

No figure is >2 pages from first discussion. No aggressive float hacks applied.

## Visual figure audit (150 dpi page renders)

| Figure | Page | Visual check |
|--------|------|--------------|
| Fig 1 | 8 | Three-panel narrative readable; no clipping |
| Fig 2 | 10 | Full 6×6 matrix; T/R/O/mask legend; `(q*,q*)` as R; V column in flowchart |
| Fig 3 | 12 | Single-channel heatmap; ΔU annotations; implementation dashboard PASS |
| Fig 4 | 16 | L10H2/L10H10 labels; CI bars; fine T/R panel; context labels legible |
| Fig 5 | 19 | Panel (b) prominent; task-specific units; gate table G0--G2 CONFIRMED, G3/G4 NOT CONFIRMED; EF-6 FAIL reads as hypothesis failure |

Local review PNGs: `review_renders/` (not committed).

## Log-derived QA (`main.log`)

| Metric | Count |
|--------|------:|
| Undefined citations | 0 |
| Undefined references | 0 |
| Duplicate labels | 0 |
| Overfull `\hbox` | 0 |
| Underfull `\hbox` | 6 |
| Underfull `\vbox` | 24 |
| Worst overfull overflow | n/a (no overfull boxes) |

**Other warnings**

- `Package hyperref Warning`: bookmark level jumps at declarations subsections (lines 25, 28 of `discussion_limitations_conclusion.tex`) — cosmetic PDF outline only
- Many `Underfull \vbox (badness 10000)` on float-heavy pages — typical Springer single-column float spacing; not corrected (no `\vspace` compression)

**Font / PDF inclusion:** no font-substitution or missing-file warnings in final pass.

## Approximate section page usage (main text)

| Block | Approx. pages |
|-------|---------------|
| Title + abstract + keywords | 1 |
| Introduction + Related Work | 2--6 |
| Problem + FALI (Sec. 3--4) | 6--11 |
| Validation (Sec. 5) | 11--13 |
| GPT-2 IOI (Sec. 6) | 14--16 |
| Pythia induction (Sec. 7) | 17--18 |
| Discussion / Limitations / Conclusion | 18--22 |
| References | 22--24 |

**Low-cost future savings (not executed here):** float whitespace around Fig. 5; Fig. 4 explicit cross-ref + tighter local placement; bibliography already compact; declaration placeholders add ~1 page when filled.

## Contribution Information Sheet

| Check | Result |
|-------|--------|
| Compile | **PASS** |
| Pages | 2 |
| Overflow | none |
| Missing metadata | none beyond intentional author placeholders |
| MLJ 1--2 page format | **satisfied** |

## Supplement

Outline-only placeholder. LaTeX infrastructure compiles (**PLACEHOLDER-BUT-COMPILES**, 1 page). Empty bibliography warning expected (`natbib`: empty `thebibliography`).

## Scientific integrity (unchanged)

| Check | Status |
|-------|--------|
| EF-1R overall PASS | unchanged |
| EF-4 fine-score gate PASS | unchanged |
| EF-5 overall gate FAIL | unchanged |
| EF-6 gate FAIL | unchanged |
| GPT-2 φ_T > 0, φ_R < 0 | unchanged |
| Pythia φ_T < 0, φ_R > 0 | unchanged |
| `pytest tests/ -q` | **205 passed** |
| `render_all.py` validation | **PASS** (empirical values/CIs/gates unchanged) |
| Experiments rerun | **no** |

## Items requiring research-lead review

1. **Page budget:** 24 pages vs. internal ~20-page target — editorial compression needed later, not layout hacks.
2. **Fig. 5 float drift:** appears on p. 19 while Discussion begins p. 18.
3. **Fig. 4:** no explicit `Figure~\ref{fig:gpt2-attention-recruitment}` in `gpt2_ioi.tex`.
4. **Submission metadata placeholders** remain in `main.tex` (authors, declarations, repository IDs).
5. **Supplement** remains outline-only for substantive writing pass.

## Review artifacts preserved locally

- `manuscript/machine_learning/main.pdf`
- `manuscript/machine_learning/review_renders/main_page-*.png` (24 pages at 150 dpi)
