# Extended FALI — Preregistration (EF-0)

Frozen **before** synthetic transformer or GPT-2 model runs.  
Mathematical specification: `docs/EXTENDED_FALI_FORMAL_SPEC.md`.

---

## Primary method hypotheses

### M1 — Five-player reconciliation

Five-player hierarchical Owen attributions (T,R,O,V,C) exactly reconcile Δ_U:

\[
\sum_i \phi_i = U(1,1,1,1,1) - U(0,0,0,0,0) = \texttt{conditional\_gain}.
\]

### M2 — SCORE block reconciliation

\[
\Phi_{\text{SCORE}} = \phi_T + \phi_R + \phi_O
\]

under fixed-order meta marginals (see formal spec §12).

### M3 — Quotient equivalence to P×V×C

The 3-group quotient game over {SCORE, V, C} with SCORE switched as a block is
equivalent to the existing Phase-3 P×V×C factorial game at matched endpoints,
provided cached patterns equal masked_softmax(cached scores). Future model check:

\[
\Phi_{\text{SCORE}} \approx \phi_{\text{pattern}},\;
\phi_V \approx \phi_{\text{value}},\;
\phi_C \approx \phi_{\text{context}}.
\]

### M4 — Original FALI special case

When utility is target attention probability, O/V/C fixed, only T and R vary, the
two-player Shapley values equal path-stable FALI target/competing attributions
(`revision_fali.fali_shapley`).

---

## Synthetic validation criteria (EF-0)

| Criterion | Threshold |
|-----------|-----------|
| Algebra efficiency residual | < 1e−12 (float64) |
| Subset vs permutation Owen max discrepancy | < 1e−12 |
| Fixed-order quotient residuals (M2) | < 1e−12 |
| Untouched additive factors | exactly zero attribution |
| Known interaction games | match analytical Owen oracle |
| M4 vs `fali_shapley` | < 1e−12 |

Implementation: `tests/test_extended_fali_math.py`.

---

## Future real-model numerical criteria (not EF-0)

| Criterion | Threshold (provisional) |
|-----------|-------------------------|
| score → pattern → z reconstruction | ≤ predefined FP32 tolerance (TBD before EF-1) |
| Endpoint Δ_U vs `conditional_gain` | ≤ predefined FP32 tolerance |
| Φ_SCORE/V/C vs P×V×C decomposition | ≤ predefined FP32 tolerance |
| Masked/future coordinates | zero drift |

---

## Explicitly not preregistered in EF-0

- Per-head empirical hypotheses (frozen after synthetic validation, before model runs).
- Extended FALI success claims on IOI or any task.
- Modal/GPU execution plans.

---

## Status

EF-0: mathematical module + synthetic tests only. **No model-level validation yet.**
