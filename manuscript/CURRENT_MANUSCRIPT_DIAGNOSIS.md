# CURRENT MANUSCRIPT DIAGNOSIS

**Branch:** `manuscript/neural-networks-template-audit`  
**Source HEAD:** `9672c029ad64dcce746b103125ff5f4e6619f842`

## Direct answers

### Which `main.tex` have we actually been editing?

**`manuscript/machine_learning/main.tex`**

Supporting evidence:

- Target of recent commits on `manuscript/reconstruction` (author metadata, figure integration, scientific finalization)
- Open IDE focus historically on `manuscript/machine_learning/main.pdf`
- README and plan docs in that directory describe the active Springer ML workflow

### What template/class does it use?

```latex
\documentclass[pdflatex,sn-mathphys-ay]{sn-jnl}
```

With vendored `sn-jnl.cls` and `sn-mathphys-ay.bst` in the same directory.

### Does that source belong to TMLR, Springer MLJ, Elsevier, or something else?

**Springer Nature / Machine Learning (`sn-jnl`).**  
It is **not** TMLR and **not** Elsevier Neural Networks.

### Where is the most recent scientific FALI prose?

| Layer | Location |
|---|---|
| Driver | `manuscript/machine_learning/main.tex` |
| Method validation | `manuscript/machine_learning/method_validation.tex` |
| GPT-2 IOI | `manuscript/machine_learning/gpt2_ioi.tex` |
| Pythia induction | `manuscript/machine_learning/pythia_induction.tex` |
| Discussion / limitations / conclusion | `manuscript/machine_learning/discussion_limitations_conclusion.tex` |
| Supplement | `manuscript/machine_learning/supplement.tex` |
| Bibliography | `manuscript/machine_learning/references.bib` |
| Figures | `manuscript/machine_learning/figures/` |

Framing: **Factorized Attention-Level Intervention** (explicitly supersedes Frozen Attention Logit Intervention).

### Are recent sections split across files from different venue workspaces?

**No.** The current Factorized manuscript is self-contained under `manuscript/machine_learning/` (Springer).  
Historical Frozen Attention prose lives separately in `paper.tex`, `archive/manuscripts/`, and `submission_tmlr/` — not mixed into the Springer `\input` chain.

### Are figure/table paths tied to obsolete manuscript folders?

Figure includes use **relative** paths under `manuscript/machine_learning/figures/` (e.g. `\input{figures/fig1}`).  
Not tied to TMLR folders. TMLR figures remain under `submission_tmlr/figures/` for the historical tree only.

### Are bibliography files duplicated or divergent?

Yes:

| Bib | Venue / era |
|---|---|
| `manuscript/machine_learning/references.bib` | CURRENT Factorized / Springer |
| `submission_tmlr/references.bib` | HISTORICAL TMLR |
| `archive/manuscripts/tmlr_rejected_references.bib` | HISTORICAL archive copy |
| `paper.tex` uses `\bibliography{references}` | expects a root `references.bib` (historical) |

These must not be silently merged without review.

### Are old Frozen Attention Logit Intervention claims still present?

Yes, but **as historical sources**, not as the Springer driver:

- `paper.tex` (root) — full Frozen Attention draft
- `archive/manuscripts/tmlr_rejected_main.tex`
- `submission_tmlr/main.tex`

The Springer `main.tex` cites the earlier preprint and states the present work supersedes it.

### Is the latest Factorized manuscript located somewhere else?

**No.** Latest Factorized content is in `manuscript/machine_learning/`.  
Gauge-sensitivity analysis results exist under `results/extended_fali/gauge_sensitivity/` on other branches / untracked local trees but are **not yet integrated into manuscript prose** on this HEAD (`9672c02`).

## Problem statement for the research lead

The repository’s active journal manuscript is built on the **Springer Machine Learning `sn-jnl` template**, while the intended target for this audit is **Neural Networks (Elsevier)**.  

**The Elsevier Neural Networks template was never added to this repository**, so the manuscript cannot currently be “fixed” by flipping a folder name. Adding/downloading the official Elsevier package is a prerequisite before establishing `manuscript/neural_networks/`.
