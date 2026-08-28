# EF-5 — Ablation Robustness: Zero vs Matched-Mean vs Matched-Resample (frozen before execution)

This document preregisters Extended FALI Phase EF-5. No new IOI hypotheses may be added
after observing alternative-scheme fine-channel values.

**Preregistration commit SHA** must appear in every `run_config.json`.

---

## 0. Primary question

Do the EF-4-confirmed fine-score assignments on canonical IOI templates remain stable
when primary-circuit damage uses distribution-matched interventions instead of zero
ablation?

| Head | Channel | Frozen direction (from EF-4) |
|------|---------|------------------------------|
| L10H2 | phi_T | > 0 |
| L10H2 | phi_R | < 0 |
| L10H10 | phi_T | > 0 |
| L10H10 | phi_R | < 0 |

No phi_O hypothesis. phi_O is descriptive only.

ZERO results from EF-4 are the frozen reference. Do not rerun zero Extended-FALI cells.

---

## 1. Three damage schemes

| Scheme | Definition |
|--------|------------|
| **ZERO** | Historical full hook_z zero ablation (EF-4 reference) |
| **MATCHED_MEAN** | Leave-one-out stratum/end-position mean of clean primary hook_z |
| **MATCHED_RESAMPLE** | Single donor prompt's clean primary hook_z (same donor for L9H6, L9H9, L10H0) |

Primary heads: L9H6, L9H9, L10H0 (unchanged).

---

## 2. Matched-mean definition

Per seed, cache clean primary hook_z. For recipient example *i* and primary head *h*:

```
replacement[i,h] = mean(clean_z[j,h] : j != i,
                         template_stratum[j] == template_stratum[i],
                         end_position[j] == end_position[i])
```

Requirements: group size ≥ 2; recipient excluded; metadata-only grouping; float32 injection.

---

## 3. Matched-resample definition

Before model outputs, build deterministic donor map *j(i)* per recipient *i*:

Eligible donor *j* must satisfy: j ≠ i; same template_stratum and end_position;
donor S ≠ recipient S; donor IO ≠ recipient IO.

Stable SHA256-seeded search. Same donor *j* used for all three primary heads.

If any recipient lacks a valid donor: **STOP** (do not relax constraints).

---

## 4. Design (unchanged from EF-4)

| Parameter | Value |
|-----------|-------|
| Model | gpt2-small, FP32, center_writing_weights |
| Dataset | StandardIOIDataset, exact Phase-4 cohort alignment |
| N per seed | 900 |
| Seeds | 42, 1, 8, 22 |
| Head panel | L9H7, L9H0, L10H2, L10H10 |
| Cells | 32 Extended-FALI per head × example |

Run **only** matched_mean and matched_resample (new GPU work). ZERO attributions from EF-4.

---

## 5. Informative-damage criterion (frozen)

Alternative scheme *s* is **INFORMATIVE** iff:

- A. 95% hierarchical CI of `damage_drop_s` entirely > 0
- B. all four seed means > 0
- C. mean(damage_drop_s) ≥ 0.25 × mean(damage_drop_zero)

If zero damage_drop is not positive with CI > 0: **STOP** (reference perturbation failure).

If alternative fails informative criterion: label **UNINFORMATIVE** — do not interpret
channel disappearance as mechanism reversal.

---

## 6. Primary robustness channels (R1–R4)

Same EF-4 directional criteria per scheme:

- hierarchical 95% CI in frozen direction
- 4/4 seed means in frozen direction
- ≥ 20/30 stratum means in frozen direction
- T0-excluded (14-cluster) CI + 4/4 seeds in frozen direction
- phi_R additionally: |mean phi_R| ≥ 0.10 × |mean Phi_SCORE|

---

## 7. Per-channel status (frozen)

| Status | Rule |
|--------|------|
| **ROBUST_SAME** | All frozen confirmation criteria pass in original direction |
| **WEAK_OR_NULL** | Original criteria fail; opposite direction NOT broadly confirmed |
| **REVERSED** | Opposite: CI entirely opposite, 4/4 seeds opposite, ≥ 20/30 strata opposite |

---

## 8. Scheme status (frozen)

| Status | Rule |
|--------|------|
| **ROBUST** | INFORMATIVE and all four channels ROBUST_SAME |
| **PARTIAL** | No REVERSED channel; uninformative or ≥1 WEAK_OR_NULL |
| **FAILED** | ≥1 REVERSED under informative damage; OR both informative and same channel fails ROBUST_SAME under BOTH alternatives |

---

## 9. EF5 ablation gate (frozen)

| Gate | Condition |
|------|-----------|
| **PASS** | matched_mean = ROBUST AND matched_resample = ROBUST |
| **PARTIAL** | no informative REVERSED channel; PASS not met |
| **FAIL** | informative REVERSED channel; OR both informative and same channel fails under BOTH |

Only **PASS** authorizes cross-setting/model generalization without research-lead review.

---

## 10. Smoke validation (before N=900)

seed=42, one template stratum, 8 examples, both matched schemes, all four heads.
Implementation validation only — do not inspect signs scientifically.

---

## 11. Outputs

```
results/extended_fali/ef5_ablation_robustness/
  smoke/
  matched_mean/seed_<seed>/
  matched_resample/seed_<seed>/
  aggregate/
```

Raw cells: `audit/extended-fali-ef5-raw` (private).

---

## 12. Historical integrity

Hash before/after: EF-4 attribution raw, EF-4 aggregate hypothesis files,
Phase-4 factorial raw. Missing required file = FAIL / NOT_AVAILABLE.

---

## 13. Prohibited

- New heads, new IOI hypotheses, R splitting, post-hoc constraint relaxation
- Cross-model/task generalization before gate PASS
- Manuscript rewrite in this phase
