# REFERENCE STYLE AUDIT — Neural Networks

**Date:** 2026-09-15  
**Branch:** `manuscript/neural-networks-content-migration`

## Journal requirement

Neural Networks Guide for Authors: citations follow **APA 7th**; reference list alphabetical then chronological.

Official Elsevier LaTeX pathway for author–year: `elsarticle` with `authoryear` + **`elsarticle-harv.bst`** (from Elsevier-maintained elsarticle 3.5 package; see `tools/elsevier-template/SOURCE.md`).

## Style tested

| Item | Value |
|---|---|
| Class | `\documentclass[preprint,12pt,authoryear]{elsarticle}` |
| BST | `elsarticle-harv.bst` (unmodified copy from official package) |
| Engine | `pdflatex` + `bibtex` via `latexmk` |
| Test dir | `manuscript/neural_networks/refstyle_test/` |

## Example rendered bibliography entries (from `test.bbl`)

**Journal / arXiv article (McGrath et al., 2023):**  
McGrath, T., Rahtz, M., Kramar, J., Mikulik, V., Legg, S., 2023. The hydra effect: … arXiv preprint arXiv:2307.15771 URL: …

**Conference (Wang et al., 2023):**  
Wang, K.R., …, 2023. Interpretability in the wild: …, in: International Conference on Learning Representations. URL: …

**Book chapter (Owen, 1977):**  
Owen, G., 1977. Values of games with a priori unions, in: Mathematical Economics and Game Theory: … Springer, New York.

**3+ authors in text:** `\citet{elhage2021mathematical}` → “Elhage et al. (2021)” (natbib authoryear).

**Same-author/same-year:** not present in the Factorized core `.bib`; style supports `a`/`b` suffixes when years collide.

## Comparison to strict APA 7 examples in the GFA

| Feature | APA 7 GFA examples | `elsarticle-harv` |
|---|---|---|
| Author–year in text | Yes | Yes (`natbib`) |
| Alphabetical list | Yes | Yes |
| Initials after surnames | Yes | Yes |
| “&” before last author | Common in APA | Comma-separated list (Elsevier Harvard) |
| DOI as https://doi.org/… | Preferred in APA | DOI/`\doi` supported when field present; many entries use `url`/`arXiv` |
| Sentence case titles | APA | BST lowercases titles (standard Harvard BST behavior) |

## Official alternative search

Elsevier elsarticle package ships only:

- `elsarticle-harv.bst` (author–year)
- `elsarticle-num.bst`
- `elsarticle-num-names.bst`

No separate “APA Style 5 / model5-names” file is in the official elsarticle 3.5 CTAN/Elsevier distribution used here. We did **not** download an unofficial GitHub APA `.bst`.

## Final recommendation

**Retain `elsarticle-harv.bst`.**

It is the supported Elsevier author–year style matching Neural Networks’ APA-like author–year requirement. Residual differences vs strict APA 7 punctuation are Elsevier production conventions, not grounds to invent a custom `.bst`.

**Decision:** proceed with `elsarticle-harv.bst` unchanged.
