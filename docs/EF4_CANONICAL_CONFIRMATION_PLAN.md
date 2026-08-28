# EF-4 — Canonical-Template Fine-Score Confirmation (frozen before execution)

This document preregisters Extended FALI Phase EF-4. No hypothesis may be added
after canonical fine-channel values are observed.

**Preregistration commit SHA** must appear in every `run_config.json`.

---

## 0. Scientific scope

Historical Phase-4 already analyzed these canonical prompts for coarse quantities:

- `conditional_gain`
- P / pattern Shapley (`Phi_SCORE` / `phi_pattern`)
- V / `phi_value`
- C / `phi_context`
- functional recruitment roles

**What is held out for EF-4:** fine-grained Extended-FALI attributions

- `phi_T`
- `phi_R`
- `phi_O`

under the five-player (T,R,O,V,C) intervention game on the canonical 30-template
distribution.

Permitted language: *canonical-template confirmation of fine-grained Extended-FALI
attributions*.

Prohibited: *independent replication of all Phase-3/4 findings*.

---

## 1. Frozen confirmatory hypotheses (four primary)

| ID | Head | Channel | Direction | Source |
|----|------|---------|-----------|--------|
| EF4-H1 | L10H2 | phi_T | > 0 | Pre-EF-2 preregistration |
| EF4-H2 | L10H2 | phi_R | < 0 | EF-3 discovery (≈ −0.343628) |
| EF4-H3 | L10H10 | phi_T | > 0 | Pre-EF-2 preregistration |
| EF4-H4 | L10H10 | phi_R | < 0 | EF-3 discovery (≈ −0.149499) |

No directional `phi_O` hypothesis. `phi_O` is descriptive only.

---

## 2. Head panel (frozen, no post-hoc selection)

| Head | Role |
|------|------|
| L9H7 | Same-layer architectural control |
| L9H0 | Same-layer architectural control |
| L10H2 | Confirmatory target |
| L10H10 | Confirmatory target |

Do not run all 12 Phase-1 candidates.

---

## 3. Design

| Parameter | Value |
|-----------|-------|
| Model | gpt2-small, FP32, center_writing_weights |
| Primary knockout | L9H6, L9H9, L10H0 |
| Dataset | StandardIOIDataset (Easy-Transformer IOI, commit e168301…) |
| Strata | 15 templates × 2 patterns (BABA/ABBA) = 30 |
| N per seed | 900 (30 prompts per stratum) |
| Seeds | 42, 1, 8, 22 |
| Cells | 32 Extended-FALI cells per head × example |
| Historical bridge | Phase-4 `factorial_raw.csv` per seed (read-only) |

Dataset alignment: exact match to `results/revision_phase4/full/seed_<seed>/dataset_metadata.csv`.

---

## 4. Primary observed estimand

`observed_canonical_balanced_mean`:

1. For each seed: mean of 30 stratum means (equal stratum weight).
2. Average four seed means equally.

Point estimates must **not** be bootstrap-replicate means (EF-3R correction applies).

---

## 5. Primary bootstrap

Use `revision_stats.hierarchical_template_cluster_bootstrap_means` (corrected Phase-4C):

- 10,000 deterministic draws
- SHA256-derived RNG seed from `(phase, head, metric, analysis_scope)`
- Sample 15 template clusters with replacement (shared across seeds)
- Retain BABA + ABBA per sampled template
- Resample within each seed × template × pattern cell independently
- Equal average across seeds (seeds not resampled)
- Percentile 95% CIs

---

## 6. Broad-template support criterion

For each EF4-H1–H4:

**A.** 95% hierarchical CI entirely in predicted direction.

**B.** All 4 seed means in predicted direction.

**C.** ≥ 20 / 30 stratum means (template × pattern) in predicted direction.

`BROAD_CONFIRMED` = A AND B AND C.

`STRONG` = BROAD_CONFIRMED AND ≥ 24 / 30 strata predicted sign.

---

## 7. Magnitude requirement (EF4-H2, EF4-H4 only)

|mean phi_R| ≥ 0.10 × |mean Phi_SCORE| using exact observed balanced means.

If |Phi_SCORE| < 1e−6: magnitude criterion fails (undefined).

For phi_T (H1/H3): no magnitude criterion beyond directional confirmation.

`CONFIRMED` (R channels) = BROAD_CONFIRMED AND magnitude pass.

For phi_T: `FINAL_CONFIRMED` = BROAD_CONFIRMED.

For phi_R: `FINAL_CONFIRMED` = BROAD_CONFIRMED AND magnitude pass.

---

## 8. Strict non-overlap sensitivity (T0 excluded)

Exclude **entire** low-level template cluster `template_id = 0` (both BABA and ABBA).

Remaining: 14 clusters, 28 strata.

Remap template IDs 1..14 → 0..13 for hierarchical bootstrap (`n_templates=14`).

Require for each primary hypothesis:

- T0-excluded CI entirely in predicted direction
- All 4 T0-excluded seed means in predicted direction

`STRICT_NONOVERLAP_PASS` = both hold.

`FINAL_CONFIRMED` additionally requires `STRICT_NONOVERLAP_PASS`.

---

## 9. Paper-level fine-score gate

| Gate | Condition |
|------|-----------|
| **PASS** | L10H2: phi_T FINAL_CONFIRMED AND phi_R FINAL_CONFIRMED **AND** L10H10: phi_T FINAL_CONFIRMED AND phi_R FINAL_CONFIRMED |
| **PARTIAL** | Exactly one head passes both channels |
| **FAIL** | Neither head passes both |

No compensatory head search if gate fails.

---

## 10. Architectural controls (L9H7, L9H0)

Across all 3600 prompts require:

- |phi_T|, |phi_R|, |phi_O|, |phi_V| ≤ 1e−6
- |phi_C − conditional_gain| ≤ 1e−6

Same-layer score/V A-vs-B invariance ≤ 1e−6.

Not stochastic confirmatory hypotheses.

---

## 11. Exploratory / descriptive only

- phi_O for L10H2/L10H10 (no directional test)
- coarse P/V/C quantities (previously observed on these cohorts)
- descriptive FALI attention fields
- leave-one-template-out diagnostics
- BABA vs ABBA heterogeneity reporting

---

## 12. Failure policy

- Any dataset misalignment → STOP
- Any implementation validation failure → STOP scientific interpretation
- Historical Phase-4 raw files must remain byte-identical
- Missing required historical file → FAIL / NOT_AVAILABLE (never null==null pass)

---

## 13. Deliverables

```
results/extended_fali/ef4_canonical/seed_<seed>/
results/extended_fali/ef4_canonical/aggregate/
```

Raw cell branch: `audit/extended-fali-ef4-raw` (private).

---

## 14. Safety / integrity

Mitigations: frozen four-head panel, frozen four channel hypotheses, exact historical
bridge, preregistered support criteria, T0-excluded sensitivity, no post-hoc head search.

Interpretability-focused; no capability scaling risk.
