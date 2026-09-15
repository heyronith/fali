# MANUSCRIPT CONFLICT REPORT

**Branch:** `manuscript/neural-networks-template-audit`

For each conflict: preferred designation only — no scientific guessing beyond provenance.

| Conflict | Sources | Designation |
|---|---|---|
| Multiple `main.tex` | `manuscript/machine_learning/main.tex`; `submission_tmlr/main.tex`; `tmlr-style-file-main/main.tex` (example) | **CURRENT SCIENCE** = ML path; TMLR = **HISTORICAL** / **TEMPLATE ONLY** |
| Title inconsistency | ML: Factorized mechanistic decomposition; TMLR/paper: “When Attention Repairs Itself” / Frozen Attention | ML = **CURRENT SCIENCE**; others = **HISTORICAL** |
| Abstract duplication / divergence | ML Factorized abstract vs TMLR/paper Frozen abstracts | ML = **CURRENT SCIENCE**; others = **HISTORICAL** |
| Method name / estimand | Factorized Attention-Level Intervention vs Frozen Attention Logit Intervention | Factorized = **CURRENT SCIENCE**; Frozen = **HISTORICAL** |
| Bibliography databases | `manuscript/machine_learning/references.bib` vs `submission_tmlr/references.bib` vs archive copy | ML bib = **CURRENT SCIENCE**; TMLR bibs = **HISTORICAL** |
| Template metadata | Springer `sn-jnl` + ML journal README vs TMLR anonymity/`tmlr.sty` vs absent Elsevier | Springer ML = **CURRENT SCIENCE** for *present* active draft; TMLR = **HISTORICAL**; Elsevier = **UNKNOWN** (missing) |
| Old L10H10-only / incomplete knockout narrative | Present in Frozen Attention drafts (`paper.tex`, TMLR) | **HISTORICAL** — do not treat as current Factorized claims |
| Figure sets | `manuscript/machine_learning/figures/` vs `submission_tmlr/figures/` | ML figures = **CURRENT SCIENCE**; TMLR figures = **HISTORICAL** |
| Author/contact metadata | Updated on ML Springer draft @ `9672c02` | ML = **CURRENT SCIENCE** |
| Gauge-sensitivity manuscript claims | Not in `main.tex` on this HEAD; results may exist off-branch | **UNKNOWN** for prose integration until authorized |

## Venue target conflict

| Statement | Status |
|---|---|
| Active edited manuscript venue | Springer Machine Learning |
| Audit-requested target venue | Neural Networks (Elsevier) |
| Elsevier template in repo | **MISSING** |

This is an organizational/venue conflict, not a scientific disagreement about experiment outcomes.
