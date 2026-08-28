# Phase 2A/3A Review Summary

**Status:** STOP after N=8 Modal smoke + package. No N=200. No manuscript rewrite.

**Branch context:** `revision/fali-validity` revision stack (does not overwrite historical `results/raw/` or Phase-1B `candidate_raw.csv`).

---

## 1. Files changed / added

| Path | Role |
|------|------|
| `experiments/revision_phase1_corrected_classification.py` | Phase-1C: SHA256-stable bootstrap seed + strong/suggestive sign-flip labels |
| `src/revision_fali.py` | Descriptive FALI (target-score / competing-score Shapley) + hybrids math |
| `src/revision_score_interventions.py` | Clone-safe score/pattern/Q/K hooks; composed hook merging |
| `experiments/revision_phase2_fali_causal.py` | Unified Phase-2A/3A pipeline (descriptive + causal panel) |
| `modal_phase2.py` | Modal L4 smoke runner (N≤8 only) |
| `tests/test_revision_phase2.py` | Algebra, mask, hook, alignment, Phase-1C regressions |
| `results/revision_phase1/full/aggregate_corrected/` | Regenerated corrected aggregates only |
| `results/revision_phase2/smoke/` | N=8 seed=42 Modal smoke outputs |

Also included: `git_diff_phase2a3a.patch`.

---

## 2. Phase-1C reproducibility fix

- Replaced `hash(head_name)` with `stable_str_hash_int` (SHA256-derived) for bootstrap RNG seeding.
- Sign-flip labels refined:
  - `strong_suppressive_to_supportive` / `suggestive_suppressive_to_supportive`
  - `strong_supportive_to_suppressive` / `suggestive_supportive_to_suppressive`
- Strong requires mean sign flip **and** CI consistency in ≥3/4 seeds.
- Suggestive requires mean/seed-mean consistency without full CI support.
- **L10H1** label after regen: `suggestive_suppressive_to_supportive` (as expected).
- No model run for Phase-1C.

---

## 3. Tests

```
python3 -m pytest tests/test_revision_phase2.py -q
→ 14 passed
```

Coverage includes: FALI algebra, softmax–margin identity, mask/hybrid semantics, output Shapley algebra, score/pattern/Q/K hook scoping + clone safety, composed hooks, Phase-1 metadata presence, historical non-delete, Phase-1C hash/sign-flip regressions.

---

## 4. Modal GPU / environment

- GPU: **NVIDIA L4**
- Image: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`
- torch `2.6.0+cu124`, transformer-lens `3.2.1`, transformers `4.57.6`
- dtype: **FP32**
- N=8, seed=42
- Device checks: CUDA model + tokens on CUDA

---

## 5–8. Smoke acceptance metrics

| Check | Result |
|-------|--------|
| Max \|FALI residual\| | **5.55e-17** (tol 1e-6) |
| Max \|causal Shapley residual\| | **0.0** (tol 1e-5) |
| Max softmax–margin error | **9.52e-8** (tol 1e-5) |
| L9H7 max \|score_row A−B\| | **0.0** |
| L9H7 max \|fali_target\| / \|Δattn\| | **0.0** / **0.0** |
| All 12 descriptive heads | PASS |
| All 6 causal-panel heads | PASS |
| No NaN/Inf (finite metrics) | PASS |

**Same-layer invariant:** L9H7 (and L9H0) score rows / FALI / Δattn are exactly zero under corrected Layer-9 `hook_z` primary knockout — confirms FALI diagnoses attention-score changes, not all forms of downstream functional recruitment.

---

## 9. Six-panel smoke outputs (means — **not for scientific inference**)

| head | F00 | F10 | F01 | F11 | causal_target | causal_competing | pattern_effect | q_eff | k_eff | qk_eff |
|------|-----|-----|-----|-----|---------------|------------------|----------------|-------|-------|--------|
| L10H2 | 2.575 | 3.155 | 2.208 | 2.906 | 0.640 | −0.308 | 0.332 | 0.319 | 0.006 | 0.330 |
| L10H1 | 2.920 | 2.943 | 2.861 | 2.906 | 0.034 | −0.048 | −0.014 | −0.014 | 0.000 | −0.014 |
| L10H10 | 2.737 | 2.987 | 2.581 | 2.906 | 0.288 | −0.118 | 0.170 | 0.173 | −0.004 | 0.170 |
| L9H7 | 2.906 | 2.906 | 2.906 | 2.906 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| L10H7 | 1.587 | 2.774 | 1.651 | 2.906 | 1.221 | 0.099 | 1.320 | 1.322 | 0.006 | 1.320 |
| L10H6 | 2.805 | 3.094 | 2.535 | 2.906 | 0.330 | −0.229 | 0.101 | 0.104 | −0.004 | 0.100 |

Descriptive smoke (selected): L10H2 and L10H10 both show large Δattn (~0.30) with large positive `fali_target`; distinguishing them is deferred to N=200.

Sign convention: `*_clean_*_effect = B_IO−S − patched_IO−S`.

---

## 10. Implementation ambiguities / notes

1. **Descriptive FALI** is attention-space only; terminology is strictly *target-score* / *competing-score* contribution (not “active routing” / “passive” / “causal recruitment”).
2. **Causal score hybrids** measure mediation via the candidate’s changed score row inside corrected B; `F11−F00` is **not** equated to the head’s full conditional effect.
3. **Pattern / Q / K** patches leave V/z and other heads in B context.
4. Optional **Q/K score Shapley** is descriptive geometry on the IO key, distinct from causal Q/K patching.
5. Modal remote image does not mount Phase-1B raw CSVs; protected-file hashes are `null` there. Locally, Phase-1B `candidate_raw.csv` mtime remains pre-smoke (`2026-08-13 14:28`) and was not written by this phase.
6. Layer-9 heads (L9H0, L9H7) both show zero score response under Layer-9 `hook_z` ablation of L9H6/L9H9 — expected parallelism.

---

## 11. Historical / Phase-1 files

- **Not modified:** Phase-1B `results/revision_phase1/full/seed_*/candidate_raw.csv` and other raw seeds.
- **Regenerated only:** `results/revision_phase1/full/aggregate_corrected/` (Phase-1C fix).
- **New only:** `results/revision_phase2/smoke/`.
- Historical `results/raw/`, `results/summary/` not written.

---

## 12. Path to ZIP

`phase2a3a_review_package.zip` (repository root)

---

## Explicit STOP

- Do **not** run N=200 yet.
- Do **not** begin manuscript rewriting.
- Full-run scientific questions (L10H2 vs L10H10, mediation fractions, L9H7 scope, etc.) are deferred until authorized.
