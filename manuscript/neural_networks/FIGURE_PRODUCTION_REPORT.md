# FIGURE PRODUCTION REPORT

**Policy:** regenerate from retained source-backed assets only; no model re-runs; scientific content unchanged.

| figure | source data | plotting script / origin | old format | final format | size | provenance | scientific content changed? |
|---|---|---|---|---|---|---|---|
| Fig1 | EF/validation figure data (archive) | `manuscript/archive/machine_learning/figure_scripts/` → archive PDF | PNG | **PDF (vector)** | 41058 B | `Fig1_provenance.json` | **NO** |
| Fig2 | method schematic data | archive figure scripts | PNG | **PDF (vector)** | 40811 B | `Fig2_provenance.json` | **NO** |
| Fig3 | synthetic/validation plot data | archive figure scripts | PNG | **PDF (vector)** | 30388 B | `Fig3_provenance.json` | **NO** |
| Fig4 | combined GPT-2 panel (archive) | archive figure scripts | PNG | **PDF (vector)** | 20476 B | `Fig4_provenance.json` | **NO** |
| Fig4a | GPT-2 attention vs recruitment panel | archival PNG (no separate vector source retained) | PNG | **PDF (PNG-wrapped)** | 249186 B | `Fig4_provenance.json` / PNG hash preserved | **NO** |
| Fig4bc | GPT-2 fine/coarse panels | archival PNG (no separate vector source retained) | PNG | **PDF (PNG-wrapped)** | 634433 B | same | **NO** |
| Fig5 | Pythia transfer figure data | archive figure scripts | PNG | **PDF (vector)** | 33989 B | `Fig5_provenance.json` | **NO** |

## Notes

- True vector PDFs for Fig1–Fig5 (combined) were copied from `manuscript/archive/machine_learning/figures/`.
- Fig4a/Fig4bc lack independent archival vector sources; attempting a fresh matplotlib redraw risked visual/scientific drift (`phi_O` key mismatch). Approved PNGs were wrapped into PDF containers for LaTeX inclusion without redrawing.
- PNGs retained alongside PDFs as archival reference assets.
- Caption edits removed internal EF/G jargon only; plotted values unchanged.
