# Phase-3D Review Summary

**Predefined verdict: PARTIAL**

All four primary mechanistic contrasts (H1–H4) replicate with seed-stable CIs, but Template A/B sign agreement fails the strict STRONG gate due to a small L10H7 `phi_context` A/B sign reversal (and tiny L10H2 `phi_value` flips). Core thesis still strongly supported; do not upgrade criteria post hoc.

---

## 1. Four runs

| Seed | Status | max \|H000 err\| | max \|Shapley res\| | interaction ID | L9H7 P/V |
|------|--------|------------------|---------------------|----------------|----------|
| 42 | **PASS** | 0 | ~1e-15 | 0 | exact 0 |
| 1 | **PASS** | 0 | ~1e-15 | 0 | exact 0 |
| 8 | **PASS** | 0 | ~1e-15 | 0 | exact 0 |
| 22 | **PASS** | 0 | ~1e-15 | 0 | exact 0 |

N=200, 100A/100B, 12 heads, Modal L4 FP32. Merged rows: 9600.

---

## 2–6. Key decompositions (seed-means)

| Head | gain | φ_P | φ_V | φ_C | pattern_effect_B_VB | cond_effect |
|------|------|-----|-----|-----|---------------------|-------------|
| **L9H7** | +0.072 | **0** | **0** | **+0.072** | 0 | +0.423 |
| **L10H2** | +0.200 | **+0.324** | +0.001 | **−0.125** | +0.305 | +0.317 |
| **L10H10** | **−0.086** | **+0.222** | +0.007 | **−0.315** | +0.114 | +0.130 |
| **L10H7** | +1.479 | **+1.462** | +0.002 | +0.015 | +1.462 | **−0.200** |
| **L10H1** | +0.070 | +0.022 | −0.001 | **+0.049** | +0.017 | +0.052 |

- **H1:** confirmed (context-only recruitment).
- **H2:** confirmed (pattern-driven +gain; context also significantly negative).
- **H3:** confirmed (pattern+ / context− cancellation → negative gain).
- **H4:** confirmed (huge pattern adaptation; still net-suppressive).
- **L10H1:** gain/context CIs >0 in 4/4; pattern CI>0 in 3/4; intact mean negative but flip not “strong” by Phase-1C CI standard alone.

---

## 7. Value adaptation

Mean |φ_V| ≈ 0.0035 vs |φ_P| ≈ 0.205 (~1.7%) and |φ_C| ≈ 0.075 (~4.6%).  
Material on this GPT-2 Small IOI setup: **no** — but do not generalize beyond this domain.

---

## 8. Template A/B

Major-head reversals on gain/φ_pattern/φ_context: **1** (L10H7 φ_context, small vs pattern).  
No reversal of L9H7 / L10H2 / L10H10 primary stories.

---

## 9. End-query vs full-sequence pattern

`pattern_effect_B_VB` vs Phase-2 end-query `causal_total`: Pearson ≈ **0.99999** across heads.  
Largest mean |diff|: L10H7 ≈ 0.024; most heads ≪ 0.003. End-query captures nearly all full-sequence pattern mediation here.

---

## 10. Verdict

**PARTIAL** (4/4 contrasts pass; template-stability gate fails strictly).

---

## 11. Unexpected

1. L10H2 φ_context significantly **negative** (−0.125) despite positive gain — pattern overcompensates.
2. L10H6 shows L10H10-like cancellation (φ_P +0.20, φ_C −0.20) with near-zero net gain.
3. End-query ≈ full-sequence pattern effect almost perfectly on this task.

---

## 12. ZIP

`phase3d_review_package.zip`
