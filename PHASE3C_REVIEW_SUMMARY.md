# Phase-3C Review Summary

**Status:** STOP after N=8 Modal smoke + package. No N=200. No manuscript edits. No descriptive-FALI rescue.

## 1. Tests

```
pytest tests/test_revision_phase3c.py
→ 9 passed
```

Covers: synthetic 3-factor Shapley, P/V→z reconstruction, full-sequence z injection / clone safety / unrelated heads, composed hooks (same-layer primary fold-in), historical presence.

## 2–5. Smoke acceptance (Modal NVIDIA L4, FP32, N=8, seed=42)

| Metric | Value |
|--------|-------|
| max z reconstruction error | **4.77e-07** |
| max \|H000 − intact\| | **5.72e-06** |
| max \|H111 − conditional\| | **6.68e-06** |
| max \|gain identity\| | **7.63e-06** |
| max Shapley residual | **4.44e-16** |

All preferred endpoint tolerances (≤1e-4) pass. All 12 heads executed.

## 6. L9H7 (smoke means — not for inference)

- φ_pattern = **0**, φ_value = **0**, φ_context = **0.0652** (= gain)
- P_A==P_B and V_A==V_B exactly (same-layer invariant PASS)
- Strong falsifiable prediction **confirmed** on smoke

L9H0 likewise: φ_P=φ_V=0, gain ≈ φ_context.

## 7. L10H2

- gain ≈ 0.351
- φ_pattern ≈ **0.318**, φ_value ≈ 0.001, φ_context ≈ 0.032
- Pattern-dominated on smoke; close to Phase-2 end-query causal_total (0.332)

## 8. L10H10

- gain ≈ 0.069 (smoke; full N=200 had negative gain — do not infer)
- φ_pattern ≈ **+0.203**, φ_value ≈ +0.009, φ_context ≈ **−0.143**
- Smoke consistent with “positive pattern adaptation partially offset by negative context”

## 9. L10H7

- intact ≈ −1.29, conditional ≈ −0.28, gain ≈ **+1.01**
- φ_pattern ≈ **+1.19**, φ_value ≈ 0.004, φ_context ≈ −0.19
- Large +gain from pattern adaptation while remaining net-suppressive

## 10. Unexpected / notable

1. Endpoint identities extremely tight (≪ 1e-4) with full-sequence z injection.
2. φ_pattern ≈ Phase-2 end-query `causal_total` for several heads on these 8 examples (L10H2 diff ≈ −0.013); larger gap for L10H7 (≈ −0.13) — full-sequence vs end-query difference is head-dependent.
3. Value factor near-zero for most panel heads on smoke; context and pattern dominate.
4. Do **not** treat N=8 as scientific confirmation of L10H10 full-run downregulation story.

## 11. Files

| Path | Role |
|------|------|
| `src/revision_factorial.py` | z reconstruct, Shapley, full hook_z injection |
| `experiments/revision_phase3c_factorial.py` | smoke/full runner |
| `modal_phase3c.py` | Modal L4 smoke |
| `tests/test_revision_phase3c.py` | unit tests |
| `results/revision_phase3c/smoke/` | smoke outputs |

## 12. ZIP

`phase3c_review_package.zip`
