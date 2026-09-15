# MLJ REFERENCE AUDIT

**Bibliography style selected:** `sn-apa` (`sn-apacite.bst`) with `\documentclass[pdflatex,sn-apa]{sn-jnl}`.

**Evidence for selection:** Machine Learning guidelines require name–year citations and encourage APA 7 author-list conventions. Official Springer `sn-jnl` option `sn-apa` is the APA-compatible style distributed with the template. Smoke-tested with BibTeX + `sn-apacite.bst` without modifying `.bst` files.

## Citation coverage

- Cited keys in main manuscript: 1
- Keys in `references.bib`: 22
- Cited but missing from bib: none
- Bib entries not cited in main: 21 (retained only if SI cites them; SI currently has limited cites)

## Known BibTeX empty-pages warnings (not invented)

Inherited from Neural Networks production bibliography for arXiv/preprint/proceedings entries without page fields (e.g., franco2025attentioncausal, grant2025divergent, nam2025causalhead, owen1977values, wang2026fewshot, wang2023ioi, zhang2024activationpatching).

## Actions

- No `.bst` modifications.
- No invented page numbers.
- DOI fields retained where present in the evidence-base bibliography.
