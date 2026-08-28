# Extended FALI — Preregistration (EF-0 / EF-0R)

Frozen **before** synthetic transformer or GPT-2 model runs.  
Mathematical specification: `docs/EXTENDED_FALI_FORMAL_SPEC.md`.

---

## EF-0R audit note (research-lead review)

**EF-0 commit `233941f` is superseded.** That implementation averaged over only **6**
permutations with frozen block order SCORE→VALUE→CONTEXT. That is **not** the standard
Owen value.

**EF-0R** restores the standard Owen value: **36** admissible permutations (all block
orders × within-block orders). Quotient consistency uses **ordinary 3-player Shapley**
on the quotient game, not sequential fixed-order meta marginals.

The superseded formulation is retained only as `_legacy_quotient_shapley_fixed_order_debug`
for audit comparison. It must not appear in primary claims.

---

## Primary method hypotheses

### M1 — Five-player reconciliation

Five-player **standard Owen** attributions (T,R,O,V,C) exactly reconcile Δ_U:

\[
\sum_i \phi_i = U(1,1,1,1,1) - U(0,0,0,0,0) = \texttt{conditional\_gain}.
\]

### M2 — SCORE block reconciliation

\[
\Phi_{\text{SCORE}} = \phi_T + \phi_R + \phi_O
\]

and **Φ_SCORE equals the ordinary Shapley value of SCORE** in the 3-player quotient
game q(K) = v(∪ members of blocks in K).

### M3 — Quotient equivalence to P×V×C

Because the quotient game is exactly the full-block SCORE/V/C intervention game, its
**ordinary Shapley decomposition** should reproduce the existing Phase-3 P×V×C Shapley
decomposition at matched real-model cells (future check; frozen before EF-1):

\[
\text{Shapley}_q(\text{SCORE}) \approx \phi_{\text{pattern}},\;
\text{Shapley}_q(\text{VALUE}) \approx \phi_{\text{value}},\;
\text{Shapley}_q(\text{CONTEXT}) \approx \phi_{\text{context}}.
\]

### M4 — Original FALI special case

When utility is target attention probability, O/V/C fixed, only T and R vary, standard
Owen values on (T,R) equal path-stable FALI target/competing attributions
(`revision_fali.fali_shapley`).

---

## Synthetic validation criteria (EF-0R)

| Criterion | Threshold |
|-----------|-----------|
| Admissible permutations | exactly 36 (unique) |
| Algebra efficiency residual | < 1e−12 (float64) |
| Subset vs permutation Owen max discrepancy | < 1e−12 (≥100 random games) |
| Quotient Shapley block residuals (M2) | < 1e−12 |
| Block-list order invariance | < 1e−12 |
| Within-block player-list order invariance | < 1e−12 |
| Known interaction games | analytical table in formal spec §19 |
| M4 vs `fali_shapley` | < 1e−12 |

Implementation: `tests/test_extended_fali_math.py`.

---

## Future real-model numerical criteria (not EF-0R)

| Criterion | Threshold (provisional) |
|-----------|-------------------------|
| score → pattern → z reconstruction | ≤ predefined FP32 tolerance (TBD before EF-1) |
| Endpoint Δ_U vs `conditional_gain` | ≤ predefined FP32 tolerance |
| Shapley_q blocks vs P×V×C decomposition | ≤ predefined FP32 tolerance |
| Masked/future coordinates | zero drift |

---

## Explicitly not preregistered in EF-0R

- Per-head empirical hypotheses (frozen after synthetic validation, before model runs).
- Extended FALI success claims on IOI or any task.
- Modal/GPU execution plans.

---

## Status

EF-0R: standard Owen mathematics + synthetic tests. **No model-level validation yet.**
