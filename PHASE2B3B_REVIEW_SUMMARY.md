# Phase 2B/3B Review Summary

**Verdict (predefined rules): PIVOT**

Across-head FALI does **not** reliably track causal score-row Shapley (target Pearson ≈ −0.35).
Leave-one-seed-out Ridge shows modest FALI gains vs simpler baselines, but that alone does not satisfy STRONG/NARROW under the predefined dual criteria. Leave-one-head-out generalization fails for all feature sets.

---

## 1. Four full runs

| Seed | N | Status | max \|FALI res\| | max \|causal res\| | L9H7 score Δ |
|------|---|--------|------------------|--------------------|--------------|
| 42 | 200 | **PASS** | ~1e-16 | 0 | 0 |
| 1 | 200 | **PASS** | ~1e-16 | 0 | 0 |
| 8 | 200 | **PASS** | ~1e-16 | 0 | 0 |
| 22 | 200 | **PASS** | ~1e-16 | 0 | 0 |

- Modal NVIDIA L4, FP32, transformers 4.57.6 / TL 3.2.1
- Dataset alignment exact vs Phase-1 `dataset_metadata.csv`
- 12 descriptive + **12 causal** heads each
- Merged rows: **9600** = 4 × 200 × 12
- Phase-1B raw / historical outputs untouched

---

## 2. FALI target ↔ causal target

- Across-head (seed-mean heads): Pearson **−0.35**, Spearman **+0.31**, sign agreement 0.80
- Per-seed across-head Pearson ≈ −0.33 to −0.36 (stable negative)
- Within-head (mean r across seeds): mixed — L10H2 ≈ +0.45; **L10H7 ≈ −0.82** (anti-aligned); L9H7 undefined (all zeros)

**Units differ** (attn-prob vs IO−S); do not equate magnitudes.

---

## 3. FALI competing ↔ causal competing

- Across-head seed-mean: Pearson **+0.25**, Spearman **+0.43**, sign agreement 0.70
- Within-head: L10H2 ≈ +0.63; **L10H7 ≈ −0.92**

---

## 4. Predictive comparison (Ridge α=1.0, StandardScaler)

**Leave-one-seed-out** (conditional_gain):

| Model | mean R² | mean MAE |
|-------|---------|----------|
| A Δattn | 0.291 | 0.340 |
| B Δmargin | 0.255 | 0.341 |
| C raw scores | 0.259 | 0.336 |
| **D FALI** | **0.336** | **0.318** |

FALI improves seed-out slightly. **Leave-one-head-out R² is largely negative for all models** (no head-level generalization of functional effects from attention features).

---

## 5. L10H2 vs L10H10

| | L10H2 | L10H10 |
|--|-------|--------|
| Δattn | 0.305 | 0.323 |
| fali_target | 0.380 | 0.387 |
| conditional_gain | **+0.200** | **−0.086** |
| causal_total | **0.305** | **0.116** |

Descriptive FALI / Δattn do **not** separate them; **causal score-row mediation does**, aligning with functional gain direction.

---

## 6. L9H7 scope boundary

Across all **800** examples: score/FALI/causal_total = **exactly 0**, while mean conditional_effect = **+0.423**, gain = **+0.072**.
FALI cannot detect same-layer functional recruitment outside attention-score adaptation.

---

## 7. L10H7 suppressor control

- conditional_effect **−0.200** (still net-suppressive)
- large |Δattn| 0.435, fali_target −0.428
- causal_total **+1.485** (score-row change mediates large IO−S shift)
- Adaptive score change ≠ useful backup

---

## 8. Predefined classification → **PIVOT**

- Does **not** reliably correspond to causal target/competing across heads (r < 0.5; target anti-correlated).
- Modest seed-out predictive edge exists but leave-one-head-out fails; rules require causal tracking for STRONG/NARROW.

Manuscript implication: substantially pivot away from “FALI detects backup recruitment”; causal score interventions and Phase-1 functional taxonomy are the stronger validated tools here.

---

## 9. Unexpected findings

1. **causal_total ≈ pattern_clean_patch ≈ qk_clean** (Pearson ≈ 1.0 across heads/seeds); **Q dominates, K ≈ 0**.
2. Across-head FALI↔causal **anti-correlation** driven largely by L10H7 (negative FALI, large positive causal_total).
3. Descriptive FALI fails the L10H2/L10H10 discrimination that causal mediation succeeds at.
4. Same-layer heads L9H0 and L9H7 both show zero score response (expected).

---

## 10. Package path

`phase2b3b_review_package.zip`

Includes: summary, patch, source/experiments, aggregate tables/reports/plots, per-seed summaries + validation, merged 9600-row CSV, hashes for bulky per-seed raws.
