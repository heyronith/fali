# FIGURE MIGRATION REPORT

**Date:** 2026-09-15  
**Source root:** `manuscript/archive/machine_learning/figures/`  
**Destination:** `manuscript/neural_networks/figures/`  
**Policy:** migrate approved assets only; do not regenerate; adapt inclusion only.

| Figure | Source path | Provenance JSON | Scientific status | Caption changed? | Concern |
|---|---|---|---|---|---|
| Fig 1 | `.../figures/Fig1.png` + `fig1.tex` | `Fig1_provenance.json` | Current Factorized FALI overview | No (wrapper unchanged) | None |
| Fig 2 | `.../figures/Fig2.png` + `fig2.tex` | `Fig2_provenance.json` | Current 32-cell game diagram | No | None |
| Fig 3 | `.../figures/Fig3.png` + `fig3.tex` | `Fig3_provenance.json` | Method validation summary | No | None |
| Fig 4a | `.../figures/Fig4a.png` + `fig4a.tex` | (via Fig4 provenance) | GPT-2 attention vs recruitment | No | None |
| Fig 4bc | `.../figures/Fig4bc.png` + `fig4bc.tex` | (via Fig4 provenance) | GPT-2 FALI channels | No | None |
| Fig 4 | `.../figures/Fig4.png` + `fig4.tex` | `Fig4_provenance.json` | Combined GPT-2 figure (archive) | No | Present for provenance; main text uses 4a/4bc |
| Fig 5 | `.../figures/Fig5.png` + `fig5.tex` | `Fig5_provenance.json` | Cross-setting reversal | No | None |

## Notes

- Inclusion uses `\includegraphics{figures/FigN.png}` relative to `manuscript/neural_networks/`.
- No scientific figure content was edited.
- Stale terminology scan deferred to compile/review pass (`MIGRATION_REVIEW.md`).
