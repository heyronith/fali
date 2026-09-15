# Machine Learning (Springer) manuscript workspace

Status: TARGET-VENUE FOUNDATION FROZEN
Target journal: **Machine Learning** (Springer Nature; journal 10994)
Working branch: `manuscript/reconstruction`

## Authoring format

The manuscript source follows the current Springer Nature `sn-jnl` LaTeX authoring template and uses the author--year math/physical-sciences option:

```tex
\documentclass[pdflatex,sn-mathphys-ay]{sn-jnl}
```

Use the **unmodified official Springer Nature journal-article template package (December 2024 version)**. Do not vendor or edit an unofficial mirror of `sn-jnl.cls`. For local compilation, place the official `sn-jnl.cls` and `sn-mathphys-ay.bst` from Springer Nature's package in this directory (or otherwise on the TeX search path).

Official template source: Springer Nature LaTeX Author Support, December 2024 journal-article template package.

## Machine Learning journal constraints frozen for this manuscript

Current journal instructions require or specify:

- a separate **1--2 page MLJ Contribution Information Sheet**;
- abstract of **150--250 words**;
- **4--6 keywords**;
- decimal headings with **no more than three levels**;
- author--year citations and an alphabetized reference list;
- editable source files at submission and revision;
- LaTeX source plus compiled PDF;
- relevant Statements and Declarations, including Competing Interests;
- a Data Availability Statement for original research;
- figures cited in numerical order with captions in the manuscript source.

For the final online submission package, Springer explicitly says **not to use subfolders** for LaTeX figures or bibliography files. The repository may remain organized during development; a later packaging step will flatten the submission directory without changing scientific source content.

## Page target

Internal target: **20 pages including references**.
Hard ceiling without research-lead approval: **22 pages including references**.

This is an internal concision target, not a journal page limit.

## Directory role

- `main.tex` -- main article skeleton and eventually the paper.
- `supplement.tex` -- technical derivations, complete diagnostics, extra tables/figures.
- `references.bib` -- verified bibliography only; no guessed metadata.
- `contribution_information_sheet.tex` -- mandatory MLJ 1--2 page contribution sheet.
- `MANUSCRIPT_PLAN.md` -- frozen section architecture and page budget.
- `FIGURE_PLAN.md` -- frozen main-figure plan and provenance requirements.
- `EVIDENCE_LEDGER.md` -- claim-to-evidence control ledger for manuscript drafting.
- `../notes/MANUSCRIPT_POSITIONING.md` -- frozen scientific positioning.
- `../notes/CLAIM_CITATION_MATRIX.md` -- novelty/precedence claim-control matrix.

## Submission-integrity rules

1. No novelty, precedence, broad generalization, or causal-interpretation sentence enters the manuscript unless it is registered in the claim/citation controls.
2. Every numerical claim must point to a committed aggregate/raw result file and analysis definition.
3. Figures must be reproducibly generated from committed scientific outputs; no decorative or generative-AI figures.
4. Failed preregistered hypotheses/gates remain visible where scientifically relevant.
5. `C` always means surrounding-network/context contribution; never a uniquely localized downstream mechanism.
6. Hybrid FALI interventions are controlled counterfactual states and may be off-manifold.
7. The journal's current policy says generative LLM use beyond copy editing should be appropriately documented in the Methods section or another suitable section. Human authors remain accountable for the final text and claims; add the required disclosure before submission based on the final workflow actually used.
