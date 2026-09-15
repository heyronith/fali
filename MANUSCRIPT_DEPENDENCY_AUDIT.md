# MANUSCRIPT DEPENDENCY AUDIT

**Manuscript root:** `manuscript/neural_networks/`  
**Rule:** every empirical main-paper statement must trace to a retained file. Deleting a chain link ⇒ KEEP.

## Figures

| Figure | Asset | Provenance JSON | Upstream results | Generating scripts |
|---|---|---|---|---|
| Fig1 | `figures/Fig1.png` | `Fig1_provenance.json` | schematic (no numeric table) | archive figure_scripts / retained PNG |
| Fig2 | `figures/Fig2.png` | `Fig2_provenance.json` | schematic 32-cell game | same |
| Fig3 | `figures/Fig3.png` | `Fig3_provenance.json` | `results/extended_fali/ef1_synthetic/validation_report.json` | EF-1 synthetic + figure script |
| Fig4a/4bc | `Fig4a.png`, `Fig4bc.png` | `Fig4_provenance.json` | `ef4_canonical/aggregate/hierarchical_bootstrap_ci.csv`, `fine_score_gate.json` | EF-4 analyze + plot |
| Fig5 | `figures/Fig5.png` | `Fig5_provenance.json` | EF-4 + EF-6 aggregate CSVs/JSONs | EF-6 analyze + plot |

## Major numerical claims → evidence

| Section | Claim | Evidence file(s) | KEEP? |
|---|---|---|---|
| Validation | Synthetic single-factor recovery | `ef1_synthetic/validation_report.json` | YES |
| Validation | Random TL endpoints | same / method_validation.tex sourced numbers | YES |
| GPT-2 | L10H2 recruitment / attention dissociation | `revision_phase1/` corrected aggregates + EF-4 CI | YES |
| GPT-2 | $\phi_T,\phi_R$ L10H2/L10H10 | `ef4_canonical/aggregate/*` | YES |
| Ablation | Matched mean/resample fine signs | `ef5_ablation_robustness/aggregate/*` | YES |
| Ablation | EF-5 overall FAIL (zero-ref) | EF-5 aggregate + `ef5_reference_diagnosis*` | YES |
| Pythia | Recruitment + sign reversal | `ef6_pythia_induction/aggregate/*` | YES |
| Gauge | Signs preserved under row-mean | `gauge_sensitivity/GAUGE_SENSITIVITY_REPORT.md` + paired summary | YES |
| Supplement | Exact tables | EF aggregates + gauge tables in `supplement/supplement.tex` | YES |

## Citation / bib

- `references.bib` — Factorized bibliography only; all keys cited in main.

## Cleanup implication

Do **not** delete any `results/extended_fali/{ef1,ef4,ef5,ef6,gauge_sensitivity}` trees, `revision_phase1`, or manuscript figure/provenance assets.
