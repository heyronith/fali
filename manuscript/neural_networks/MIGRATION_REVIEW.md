# MIGRATION REVIEW — Neural Networks vs archived Springer Factorized FALI

**Date:** 2026-09-15  
**NN PDF:** `manuscript/neural_networks/main.pdf` (34 pages, elsarticle preprint 12pt)  
**Springer PDF:** `manuscript/archive/machine_learning/main.pdf` (24 pages, sn-jnl)  
**Scope:** migration fidelity only (not scientific quality judgment).

## Scientific content preserved

- Factorized FALI estimand, $T/R/O/V/C$ game, Owen coarse–fine reconciliation preserved.
- Negative-result framing retained: conditional recruitment generalizes; preregistered fine mechanism does not.
- GPT-2: $T>0$, $R<0$; Pythia: $T<0$, $R>0$.
- EF-5: matched variants preserve fine channels; overall gate FAIL on zero-reference criterion.
- Gauge sensitivity integrated with bounded claim A only.
- Novelty non-claims and “to our knowledge” priority language retained.
- Context factor $C$ remains surrounding-network/context residual (not “downstream sensitivity”).

## Section hierarchy

| Springer | Neural Networks |
|---|---|
| Introduction | Introduction |
| Related Work | Related Work |
| Problem Formulation | Problem Formulation |
| Factorized Attention-Level Intervention | Factorized Attention-Level Intervention |
| Method Validation | Method Validation |
| GPT-2 IOI (incl. EF-5 subsection) | GPT-2 IOI + separate Ablation and Gauge Robustness |
| Pythia induction | Pythia Induction Transfer |
| Discussion / Limitations / Conclusion | Same |
| Statements and Declarations | Elsevier front/back matter (CRediT, competing interest, etc.) |

## Density / production

- Preprint 12pt + elsarticle front matter yields **34 pages** vs Springer **24 pages** — expected density drop, not content inflation.
- Equations render correctly; no manual negative spacing introduced.
- Figures migrate as PNG wrappers; sizing uses `\textwidth` / `0.95\textwidth` as in archive.
- Bibliography: `elsarticle-harv` author–year (APA-like). Empty-pages BibTeX warnings for some entries (field metadata), not unresolved cites.
- Front matter: verified authors/affiliations from Factorized source; abstract 181 words; 6 keywords; 5 highlights ≤85 chars.

## Obvious production issues

- 17 overfull `\hbox` warnings (mostly long unbreakable paths/words in Limitations and appendix pointer); no underfull hboxes.
- Supplement compiles separately (7 pages); no in-supplement `\cite` (bibliography omitted to avoid empty `thebibliography`).
- Font size substitution warning (up to 4.5pt) from elsarticle/default fonts — template-typical.
- Figure float placement pushes some figures later in the preprint layout (production polish deferred).

## Stale scientific language

- No Frozen Attention Logit Intervention scientific claims in migrated body.
- Historical preprint mention in Related Work retained as supersession statement (present in Factorized source; not restored Frozen claims).
- No “FALI passed the ablation-robustness experiment” wording.

## Unsupported numerical statements

None found relative to archive + gauge report `4313519`.
