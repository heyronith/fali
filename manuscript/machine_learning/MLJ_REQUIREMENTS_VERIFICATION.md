# MLJ Requirements Verification

**Date:** 2026-09-15  
**Sources consulted:** Machine Learning (Springer) Submission Guidelines (journal 10994); Springer Nature LaTeX author support (December 2024 journal-article template package); research-lead `MLJ_REQUIREMENTS_AUDIT.md`.

## Confirmed requirements

| Requirement | Status | Notes |
|---|---|---|
| Mandatory 1–2 page MLJ Contribution Information Sheet (4 questions) | **Confirmed** | Incomplete sheets returned without review |
| Springer Nature LaTeX journal-article template | **Confirmed** | Official package marketed as December 2024; class file used here is `sn-jnl.cls` (`ProvidesClass` date 2019/11/18 v0.1 in distributed package; template user-manual notes Version 2.1 April 2023 in `sn-article.tex`) |
| Abstract 150–250 words | **Confirmed** | |
| Keywords 4–6 | **Confirmed** | Using exactly 6 research-lead keywords |
| Title page: authors, affiliations, corresponding email, ORCID if available | **Confirmed** | ORCID fields not invented (none verified) |
| Statements and Declarations | **Confirmed** | Funding, Competing Interests, Data/Code, Author Contributions |
| Name–year citations; APA 7 author-list conventions encouraged | **Confirmed** | Selected `sn-apa` / `sn-apacite.bst` |
| Figures: EPS preferred vector; lettering ~8–12 pt; lowercase panels; captions in text; 84/174 mm widths | **Confirmed** | Research-lead figures used unchanged |
| No subfolders in LaTeX upload ZIP | **Confirmed** | Flattened staging for submission ZIP |
| Supplementary Information separate; independent numbering | **Confirmed** | |
| Substantive LLM disclosure in Methods | **Confirmed** | Research-lead paragraph placed at end of Methods (`method_validation.tex`) |

## Bibliography style selection

**Selected:** `\documentclass[pdflatex,sn-apa]{sn-jnl}` with official `sn-apacite.bst`.

**Why:** Machine Learning submission guidelines require name–year citations and encourage APA 7 author-list conventions. The official `sn-jnl` option `sn-apa` loads `apacite` + `sn-apacite.bst`, which is the Springer-distributed APA-compatible style. Rendered smoke test with required preamble packages (`manyfoot`, `xcolor`, …) produced author–year citations and APA-style reference list entries without modifying any `.bst` file.

**Not selected:** `sn-mathphys-ay` (author–year but not the APA-targeted option); numbered styles (incompatible with MLJ name–year requirement).

## Author-confirmation flags

- ORCID IDs: not present; do not invent.
- Postal street addresses: template uses city/country; street/postcode not verified.
- Additional prior-publication overlap beyond cited Frozen Attention preprint: CIS retains explicit author-confirmation sentence.
- Data/code persistent DOI/URL: future-release wording retained pending archival identifiers.
