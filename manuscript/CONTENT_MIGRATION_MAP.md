# CONTENT MIGRATION MAP

Target venue: **Neural Networks (Elsevier)** — workspace **not yet creatable** (template missing).  
Current scientifically valid Factorized sources live under `manuscript/machine_learning/`.

| Target Neural Networks section | Current source file | Branch/path | Scientific status | Needs migration? | Needs rewriting? | Notes |
|---|---|---|---|---|---|---|
| Title | `manuscript/machine_learning/main.tex` | `manuscript/reconstruction` @ `9672c02` | CURRENT SCIENCE (Factorized) | YES (when NN template exists) | Venue front-matter only | `FALI: Mechanistic Decomposition of Conditional Attention-Head Importance` |
| Abstract | `manuscript/machine_learning/main.tex` | same | CURRENT SCIENCE | YES | Possibly for Elsevier length/keywords | Factorized framing |
| Introduction | `manuscript/machine_learning/main.tex` | same | CURRENT SCIENCE | YES | Possibly light copy-edit | Includes supersession note vs Frozen Attention |
| Related Work | `manuscript/machine_learning/main.tex` | same | CURRENT SCIENCE | YES | Review for venue | Embedded in main |
| Problem Formulation / Method | `manuscript/machine_learning/main.tex` § Factorized… | same | CURRENT SCIENCE | YES | No scientific rewrite in this audit | |
| Method Validation | `manuscript/machine_learning/method_validation.tex` | same | CURRENT SCIENCE | YES | No | |
| GPT-2 IOI | `manuscript/machine_learning/gpt2_ioi.tex` | same | CURRENT SCIENCE | YES | No | |
| Robustness / ablation notes | ML main + ledger docs; EF-5 results in `results/` | same | CURRENT SCIENCE (experiments) | YES later | Possibly | Not a separate `.tex` section name |
| Pythia induction | `manuscript/machine_learning/pythia_induction.tex` | same | CURRENT SCIENCE | YES | No | |
| Gauge-sensitivity analysis | **Not in manuscript prose on this HEAD** | results on other branch / untracked | CURRENT SCIENCE (results exist elsewhere) | YES later | YES (new section) | Do not invent prose here |
| Discussion | `manuscript/machine_learning/discussion_limitations_conclusion.tex` | same | CURRENT SCIENCE | YES | No | |
| Limitations | same file | same | CURRENT SCIENCE | YES | No | |
| Conclusion | same file | same | CURRENT SCIENCE | YES | No | |
| References | `manuscript/machine_learning/references.bib` | same | CURRENT SCIENCE | YES | Bib style change for Elsevier | |
| Figures/tables | `manuscript/machine_learning/figures/` | same | CURRENT SCIENCE | YES | Path/wrapper adapt | Do not regenerate in this audit |
| Supplement | `manuscript/machine_learning/supplement.tex` | same | CURRENT SCIENCE | YES | Elsevier appendix conventions TBD | |
| — | `paper.tex` | repo root | HISTORICAL Frozen Attention | NO (as primary) | N/A | Obsolete framing |
| — | `archive/manuscripts/tmlr_rejected_main.tex` | archive | HISTORICAL Frozen Attention | NO | N/A | Flag: Frozen Attention Logit Intervention |
| — | `submission_tmlr/main.tex` | submission_tmlr | HISTORICAL TMLR | NO | N/A | Duplicate historical |

## Obsolete-framing flag

Any source whose central scientific framing is still **Frozen Attention Logit Intervention** is **HISTORICAL/obsolete** for the active Factorized paper and must not be silently migrated as current claims.
