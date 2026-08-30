# Manuscript Compile Audit

**Branch:** `manuscript/reconstruction`  
**Audit patch base:** `e751bf9289dfd588e1c13f86d6650ef75684e585`  
**Status:** **PASS**

## Final audit-patch recompile (current)

Applied mechanical patches from `FINAL_SCIENTIFIC_AUDIT.md` only (metric wording, novelty citations, Evidence Ledger provenance, MIB author metadata). No layout hacks.

| Metric | Pre-patch (`f61ed4d`) | Post-patch (current) | Δ |
|--------|----------------------:|---------------------:|--:|
| Total pages | 20 | **21** | +1 |
| References begin | 19 | **18** | −1 |
| Bibliography pages | 2 (19–20) | **3 (18–20)** | +1 |
| Fig 1–5 pages | 8, 10, 12, 14, 16 | 8, 10, 12, 14, 16 | 0 |
| Undefined citations | 0 | **0** | 0 |
| Undefined references | 0 | **0** | 0 |
| Duplicate labels | 0 | **0** | 0 |
| Overfull `\hbox` | 0 | **0** | 0 |
| Underfull `\hbox` | 3 | **3** | 0 |
| Underfull `\vbox` | 20 | **21** | +1 |

**Page-budget classification:** **ACCEPTABLE** (21 pages; internal target ≤20, acceptable band 21–22)

## Post-compression recompile (prior)

Recompiled after research-lead scientific compression pass. No further prose edits, layout hacks, or margin/font changes applied in this step.

| Metric | Baseline (`dc4fa55`) | Current (`c0f32af` + recompile) | Δ |
|--------|---------------------:|----------------------------------:|--:|
| Total pages | 24 | **20** | **−4** |
| References begin | 22 | **19** | −3 |
| Bibliography pages | 3 (22–24) | **2 (19–20)** | −1 |
| Fig 1 page | 8 | 8 | 0 |
| Fig 2 page | 10 | 10 | 0 |
| Fig 3 page | 12 | 12 | 0 |
| Fig 4 page | 16 | **14** | −2 |
| Fig 5 page | 19 | **16** | −3 |
| Overfull `\hbox` | 0 | 0 | 0 |
| Underfull `\hbox` | 6 | **3** | −3 |
| Underfull `\vbox` | 24 | **20** | −4 |

**Page-budget classification:** **ON TARGET** (≤20 pages including references)

## Official Springer Nature template provenance

Downloaded from Springer Nature LaTeX Author Support, **journal article template package (December 2024 version)**:

- **Page:** https://www.springernature.com/gp/authors/campaigns/latex-author-support/see-where-our-services-will-take-you/18782940
- **CMS URL:** `https://resource-cms.springernature.com/springer-cms/rest/v1/content/18782940/data/v12`
- **Archive filename:** `sn-journal-template.zip`
- **Archive SHA256:** `812e76dcaa9c28dc1bff1fb6065d51729b67d4ea140552a05088317414a3ecae`
- **`sn-jnl.cls` SHA256:** `36d0c3273a59d48dc6a9c7b080dfa1ec50dc10229d8751568d1f2e490ffa5ecc`
- **`sn-mathphys-ay.bst` SHA256:** `6a2306683997db76b5e58190acea4dac360c742044114f6a923a4380fbab9008`

Unmodified copies in `manuscript/machine_learning/`.

## LaTeX toolchain

User-local TinyTeX at `/Users/ronny/fali-tinytex`.

| Tool | Version |
|------|---------|
| `pdflatex` | pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026) |
| `latexmk` | John Collins, 9 March 2026. Version 4.88 |
| `bibtex` | BibTeX 0.99e (TeX Live 2026) |

## Compile command

```bash
export PATH="/Users/ronny/fali-tinytex/bin/universal-darwin:$PATH"
cd manuscript/machine_learning
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Compile result

| Document | Status | Output | Pages |
|----------|--------|--------|------:|
| `main.tex` | **PASS** | `main.pdf` | **21** |
| `contribution_information_sheet.tex` | **PASS** | `contribution_information_sheet.pdf` | 2 |

## Bibliography (current)

| Check | Result |
|-------|--------|
| `references.bib` found | yes |
| Style `sn-mathphys-ay.bst` used | yes |
| BibTeX completed | yes |
| Author--year citations render | yes |
| Bibliography in PDF | yes (pages **18–20**) |
| References begin | **page 18** |
| Bibliography page count | **3** |
| `?` / `??` citations | none |

## Page budget (current)

| Metric | Value |
|--------|------:|
| Total `main.pdf` pages | **21** |
| References begin | **page 18** |
| Bibliography page count | **3** |
| Approx. scientific main text (title through conclusion) | **~17 pages** |

## Figure pages and float placement (current)

| Figure | Label | Figure page | Baseline page | Notes |
|--------|-------|------------:|--------------:|-------|
| Fig 1 | `fig:fali-overview` | 8 | 8 | unchanged |
| Fig 2 | `fig:fali-game` | 10 | 10 | unchanged |
| Fig 3 | `fig:validation` | 12 | 12 | unchanged |
| Fig 4 | `fig:gpt2-attention-recruitment` | **14** | 16 | closer to GPT-2 Sec. 6; explicit `\ref` in Sec. 6.3 |
| Fig 5 | `fig:cross-setting-reversal` | **16** | 19 | adjacent to Pythia Sec. 7.2; Discussion Sec. 8 begins same page |

## Visual layout audit (150 dpi, post-compression)

| Check | Result |
|-------|--------|
| Paragraph/heading orphans from compression | none observed |
| Fig 4 near GPT-2 discussion | **improved** (p. 14 vs. p. 16 baseline) |
| Fig 5 near Pythia result | **improved** (p. 16 vs. p. 19 baseline) |
| Discussion around Fig 5 | acceptable — Sec. 8 begins below Fig 5 on p. 16 |
| Figure text readability | all five figures legible at manuscript size |
| New large blank regions | none; modest trailing whitespace on p. 17 (Limitations lead-in) |

## Log-derived QA (`main.log`, current)

| Metric | Count |
|--------|------:|
| Undefined citations | 0 |
| Undefined references | 0 |
| Duplicate labels | 0 |
| Overfull `\hbox` | 0 |
| Underfull `\hbox` | 3 |
| Underfull `\vbox` | 21 |
| Worst overfull overflow | n/a |

**Other warnings**

- 2× `Package hyperref Warning` (bookmark level jumps at declaration subsections in `main.tex`) — cosmetic PDF outline only
- Residual underfull vboxes on float-heavy pages — unchanged in character, reduced in count

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
| `render_all.py` validation | **PASS** |
| Experiments rerun | **no** |

## Remaining items for research-lead review

1. **Declaration placeholders** on p. 20 (empty subsection headings before References) will expand when filled.
2. **Supplement** remains outline-only.
3. **Author/metadata placeholders** in `main.tex` unchanged.

## Review artifacts (local)

- `manuscript/machine_learning/main.pdf`
- `manuscript/machine_learning/review_renders/main_page-*.png` (21 pages at 150 dpi; not committed)

## Historical note — pre-compression compile (`dc4fa55`)

First successful Springer compile before scientific compression: 24 pages, references pp. 22–24, page-budget concern. LaTeX plumbing fix only: removed undefined `\jyear{2026}` from `main.tex`.
