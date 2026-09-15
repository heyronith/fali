# COMPILE REPORT — Neural Networks content migration

**Date:** 2026-09-15  
**Branch:** `manuscript/neural-networks-content-migration`  
**Engine:** TinyTeX `pdflatex` + `bibtex` via `latexmk`

## Main manuscript (`main.tex`)

| Metric | Value |
|---|---|
| Compile | **SUCCESS** |
| Pages | **34** |
| Abstract words | **181** (≤250) |
| Keywords | **6** |
| Highlights | 5 items; all ≤85 characters (checked by `tools/check_highlights.py`) |
| Unresolved citations | **0** |
| Unresolved references | **0** |
| Duplicate labels | **0** |
| Missing figures | **0** |
| Overfull `\hbox` | **17** |
| Underfull `\hbox` | **0** |

### Bibliography warnings (`main.blg`)

`Warning--empty pages` for: franco2025attentioncausal, grant2025divergent, nam2025causalhead, owen1977values, wang2026fewshot, wang2023ioi, zhang2024activationpatching.

All 22 `.bib` keys are cited; no orphan entries; no unresolved keys.

### Highlights character counts

| # | Chars | Text |
|---|---|---|
| H1 | 74 | FALI decomposes conditional causal usefulness of selected attention heads. |
| H2 | 71 | GPT-2 IOI shows target-positive and rival-negative fine-score channels. |
| H3 | 68 | Pythia induction reverses the preregistered target/rival fine signs. |
| H4 | 65 | Conditional recruitment generalizes; the fine mechanism does not. |
| H5 | 66 | FALI discriminates mechanisms under an explicit intervention game. |

## Supplement (`supplement/supplement.tex`)

| Metric | Value |
|---|---|
| Compile | **SUCCESS** |
| Pages | **7** |
| Bibliography | omitted (no `\cite` in supplement body) |

## Style

- Class: `elsarticle` preprint, 12pt, `authoryear`
- BST: `elsarticle-harv.bst` (unmodified; see `REFERENCE_STYLE_AUDIT.md`)
