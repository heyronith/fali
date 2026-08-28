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

---

## EF-1 — Synthetic ground-truth + tiny random transformer (frozen before implementation)

EF-1 validates the score → pattern → value → z → causal-utility pipeline against
standard Owen mathematics **before** GPT-2 / IOI / Modal. Not scientific evidence.

### Level A — Hand-constructed tensor micro-system (float64)

| Criterion | Threshold |
|-----------|-----------|
| T/R/O partition disjointness & coverage | exact; masked coords excluded |
| Masked attention probability | ≤ 1e−8 on invalid coords |
| Variable valid seq lengths / padding | included in partition tests |
| Five single-channel cases (T,R,O,V,C) | intended factor = Δ_U; others < 1e−12 |
| T×C pipeline interaction | φ_T = φ_C = Δ_U/2; others < 1e−12 |
| Endpoint reconstruction (all-A / all-B) | max error < 1e−12 |
| Δ_U = U(11111)−U(00000) vs direct conditional_gain | max error < 1e−12 |
| Quotient bridge vs `three_factor_shapley` | max error < 1e−12 |

### Level B — Tiny random HookedTransformer (CPU, FP32, no download)

Architecture (fixed): `n_layers=2`, `n_heads=2`, `d_model=32`, `d_head=16`,
`d_mlp=64`, `n_ctx=8`, `d_vocab=48`, seed **4242**.

| Criterion | Threshold |
|-----------|-----------|
| Score → pattern vs cached hook_pattern (A/B) | max error ≤ **2e−6** |
| z = P @ V vs cached hook_z (A/B) | max error ≤ **2e−6** |
| Endpoint U(00000) vs intact_effect | ≤ **1e−5** |
| Endpoint U(11111) vs conditional_effect | ≤ **1e−5** |
| U(11111)−U(00000) vs conditional_gain | ≤ **1e−5** |
| No-live-candidate-leak (one hybrid cell) | ≤ **1e−6** |
| Quotient bridge vs `three_factor_shapley` | max error ≤ **1e−5** |

PRIMARY = layer 0 head 0; CANDIDATE = layer 1 head 0. Metric = final-position logit
difference between two fixed vocabulary indices. No IOI semantics.

### Failure policy

If **any** preregistered primary criterion fails: STOP, report cause, do not patch
thresholds post hoc, do not proceed to GPT-2.

### Deliverables

- `src/extended_fali_interventions.py`
- `experiments/extended_fali_ef1_synthetic.py`
- `results/extended_fali/ef1_synthetic/validation_report.json`
- `tests/test_extended_fali_interventions.py`
- `tests/test_extended_fali_synthetic_transformer.py`

---

## EF-1R — Batch-safe partitions + independent endpoint validation (frozen before implementation)

EF-1R was required by research-lead code review because EF-1 used a batch-shared
target partition and endpoint checks were not fully independent. No
pretrained-model experiment had begun.

### Changes (implementation repair, not new science)

| Item | Requirement |
|------|-------------|
| Partition masks | `[B,Q,K]` per example; `build_batched_score_partition(...)` |
| Validity | `valid[b,q,k]` iff `q < valid_length[b]` AND `k <= q` |
| Level B targets | Per-example `target_key[b]` from argmax final-row pattern mass |
| Independent endpoints | Four direct passes (A/B × present/zero); compare to U(00000)/U(11111) |
| Endpoint activation | All-A/all-B hybrid z matches cached present-run z (≤ 2e−6) |
| Injection scope | Non-candidate heads unchanged at hook_z (≤ 1e−7) |
| Live leak | Separate V, Q, Q+V perturbations (≤ 1e−6 each) |
| Level A batch test | 3 examples with distinct final_query/target_key/valid_length |

Thresholds unchanged from EF-1 Level A/B tables above.

### Failure policy

Same as EF-1: any primary criterion fails → STOP; do not patch thresholds post hoc.

