# Manuscript Compile Audit

**Branch:** `manuscript/reconstruction`  
**Integration commit base:** `e7c476c22e51e6c36f8193c608ac9ed02c53bbd5`  
**Status:** **BLOCKED_BY_TEMPLATE_ASSETS**

## Summary

Approved Figures 1–5 are integrated into `main.tex`, `method_validation.tex`, `gpt2_ioi.tex`, and `pythia_induction.tex` via `figures/fig1.tex` … `figures/fig5.tex` using vector PDF sources. Scientific prose and captions were preserved; only minimal cross-reference inserts were added.

Full Springer manuscript compilation could **not** be executed in this environment.

## Compiler / tool availability

| Tool | Status |
|------|--------|
| `pdflatex` | not found |
| `latexmk` | not found |
| `bibtex` / `biber` | not tested (LaTeX unavailable) |
| `tectonic` | not found on PATH |

## Missing official Springer Nature template assets

The manuscript requires the **unmodified December 2024** Springer Nature journal-article package. The following files are **not present** locally or on the TeX search path:

- `sn-jnl.cls`
- `sn-mathphys-ay.bst`

Expected placement per `manuscript/machine_learning/README.md`: in `manuscript/machine_learning/` or otherwise on the TeX search path.

No substitute document class or style file was introduced.

## Intended compile commands

From `manuscript/machine_learning/` once assets are installed:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error contribution_information_sheet.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

## Figure integration map

| Figure | Include file | Inserted in | Label |
|--------|--------------|-------------|-------|
| Fig 1 | `figures/fig1.tex` | `main.tex` after Sec. mechanism-resolution objective | `fig:fali-overview` |
| Fig 2 | `figures/fig2.tex` | `main.tex` after counterfactual-game warning | `fig:fali-game` |
| Fig 3 | `figures/fig3.tex` | `method_validation.tex` after opening paragraph | `fig:validation` |
| Fig 4 | `figures/fig4.tex` | `gpt2_ioi.tex` after canonical mechanism subsection | `fig:gpt2-attention-recruitment` |
| Fig 5 | `figures/fig5.tex` | `pythia_induction.tex` after transfer-results paragraph | `fig:cross-setting-reversal` |

All figures use `\includegraphics{figures/FigN.pdf}` inside `figure*`.

## Compile outputs

| Artifact | Status |
|----------|--------|
| `main.pdf` | **not generated** |
| `contribution_information_sheet.pdf` | **not generated** |
| `supplement.pdf` | **not generated** |

## Page / float QA

Not applicable until compile succeeds.

Internal review target remains ~20 pages including references; hard review threshold >22 pages.

## Reference / label QA

Automated log scan deferred until compile succeeds. Manual source inspection:

- Five unique figure labels present in include files.
- Five minimal prose references added (`Figure~\ref{...}`).
- No duplicate figure labels introduced.
- Bibliography file `references.bib` unchanged.

## Scientific integrity (unchanged)

| Check | Status |
|-------|--------|
| EF-1R overall PASS | unchanged |
| EF-4 fine-score gate PASS | unchanged |
| EF-5 overall gate FAIL | unchanged |
| EF-6 gate FAIL | unchanged |
| GPT-2 φ_T > 0, φ_R < 0 | unchanged |
| Pythia φ_T < 0, φ_R > 0 | unchanged |
| Figure extracted values vs first draft | verified by tests |

## Missing submission metadata (pre-existing)

- Author metadata placeholders in `main.tex`
- Competing interests / funding / data / code / contributions / AI disclosure placeholders
- Final repository identifiers

## Supplement status

`supplement.tex` is a placeholder outline only; compilation blocked by missing `sn-jnl.cls` even if LaTeX were available.

## Contribution sheet status

`contribution_information_sheet.tex` uses standard `article` class and should compile independently once a LaTeX distribution is installed; not attempted here because `pdflatex` is unavailable.

## Next step

Install official Springer Nature `sn-jnl` December 2024 assets into `manuscript/machine_learning/`, install a LaTeX distribution, rerun `latexmk`, then update this audit with page counts, float locations, and log-derived warning counts.
