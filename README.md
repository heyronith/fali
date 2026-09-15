# FALI — Factorized Attention-Level Intervention

Canonical research repository for the **current** Factorized FALI paper submitted toward *Neural Networks* (Elsevier).

**Active manuscript:** [`manuscript/neural_networks/`](manuscript/neural_networks/)

**Cleanup baseline branch:** `maintenance/current-fali-cleanup`  
**Safety tag:** `pre-current-fali-cleanup`

---

## Current science (Factorized FALI)

FALI = **Factorized Attention-Level Intervention**: an output-grounded counterfactual game that decomposes the cross-state change in a selected attention head’s causal usefulness into \(T/R/O/V/C\) with standard Owen attribution.

Empirical anchors:

| Result | Location |
|--------|----------|
| Method validation (EF-1R) | `results/extended_fali/ef1_synthetic/` |
| GPT-2 canonical fine scores (EF-4) | `results/extended_fali/ef4_canonical/` |
| Ablation robustness / EF-5 FAIL gate | `results/extended_fali/ef5_ablation_robustness/` |
| Pythia transfer / EF-6 FAIL | `results/extended_fali/ef6_pythia_induction/` |
| Gauge sensitivity (signs preserved) | `results/extended_fali/gauge_sensitivity/` |

See [`CURRENT_EXPERIMENTS.md`](CURRENT_EXPERIMENTS.md) and [`results/CURRENT_PAPER_RESULTS_MANIFEST.json`](results/CURRENT_PAPER_RESULTS_MANIFEST.json).

Central empirical framing: **conditional recruitment generalizes; the preregistered fine mechanism does not** (GPT-2 \(T>0,R<0\); Pythia \(T<0,R>0\)).

---

## Historical material

Frozen Attention Logit Intervention (old FALI name) is **provenance only**:

- [`archive/historical_fali/`](archive/historical_fali/)
- [`archive/manuscripts/tmlr_rejected_*`](archive/manuscripts/)

Do not treat Frozen-era scripts/results as current evidence.

---

## Layout

```text
src/                 # Factorized FALI library
experiments/         # Current-paper experiment drivers
tests/
results/             # Current evidence (+ indexed local raw)
manuscript/
  neural_networks/   # Active Elsevier manuscript
  archive/           # Archived Factorized Springer source
tools/
  elsevier-template/ # Pristine journal template
archive/
  historical_fali/   # Minimal Frozen / TMLR provenance
docs/                # Specs / preregistration plans
preregistration/
```

---

## Reproduce

```bash
python -m pytest tests/ -q
# manuscript (requires TeX)
cd manuscript/neural_networks && latexmk -pdf -interaction=nonstopmode main.tex
```

Large cell-level tensors are gitignored and indexed in [`results/RAW_DATA_INDEX.json`](results/RAW_DATA_INDEX.json).
