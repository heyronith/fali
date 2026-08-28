# PHASE4A Review Summary — Canonical IOI Template Expansion

**Status:** PASS (smoke / execution validation only)  
**Date:** 2026-08-14  
**Scope:** Replace 2-template limitation with Wang/Redwood 15×2=30 strata; N=60 Modal factorial smoke.  
**Not in scope:** N=900 full run, scientific interpretation of N=60 template effects, manuscript rewrite, other models.

## Frozen Phase-3 hypotheses (unchanged)

| ID | Head | Prediction |
|----|------|------------|
| H1 | L9H7 | φ_P ≈ 0, φ_V ≈ 0, φ_C > 0 |
| H2 | L10H2 | conditional_gain > 0, φ_P > 0 (context potentially negative) |
| H3 | L10H10 | φ_P > 0, φ_C < 0, cancellation / negative gain |
| H4 | L10H7 | φ_P large positive, conditional_effect < 0 |
| Secondary | L10H1 | suggestive context-led recruitment |

These were **not** redefined after seeing broad-template smoke data.

## 1. Canonical source / provenance

- **Repo:** https://github.com/redwoodresearch/Easy-Transformer  
- **File:** `easy_transformer/ioi_dataset.py`  
- **Commit SHA:** `e168301aa05154a14d52e5b6f98b6518200f5b1f`  
- **Retrieval date (UTC):** 2026-08-14  
- **Paper:** arXiv:2211.00593 Appendix E (same 15 BABA strings)  
- **Local mirror SHA256:** `94a72de90e426382322fd690669024a0d2d4bd72a45ed9ce820a99ae9173d231`  
- **Provenance JSON:** `results/revision_phase4/provenance/standard_ioi_provenance.json`

ABBA templates are produced with the **exact** Easy-Transformer first-clause `[B]↔[A]` swap (not hand-paraphrased).

## 2–4. Template / name inventory

| Item | Count |
|------|------:|
| Low-level BABA templates | 15 |
| BABA+ABBA strata | 30 |
| Names (source / usable single-token GPT-2) | 99 / 99 |
| Places / objects usable | 8 / 8 |
| Tokenization rejects | **0** (all canonical names/places/objects single-token) |

## 5. N=60 balance validation

- Seed 42; **exactly 2 examples × 30 strata = 60**
- All 15 template IDs present; both BABA and ABBA for every template
- All token/index assertions passed (`StandardIOIDataset.validate()`)
- Metadata: `results/revision_phase4/smoke/dataset_metadata.csv`

## Legacy overlap (exact only)

- Legacy template **B** is an **exact** prefix match to canonical BABA `T00`
- Legacy template **A** has **no** exact match to any canonical BABA prefix
- Report: `results/revision_phase4/smoke/legacy_template_overlap_report.md`

## 6. Tests

`pytest tests/test_revision_phase4.py` → **12 passed**

Coverage: 15 templates, ABBA transform, balanced generation, determinism, positions/padding, single-token validation, metadata columns, within-stratum + template-cluster bootstrap (BABA+ABBA paired), SHA256 seeds, no historical overwrite targets.

## 7–9. Factorial smoke (Modal NVIDIA L4, FP32)

- Command: `modal run modal_phase4.py --mode smoke`
- Heads: all 12 Phase-1 candidates; full-head P×V×C (Phase-3D machinery)
- **ACCEPTANCE: PASS**

| Metric | Value |
|--------|------:|
| Max \|H000 − intact\| | 0 |
| Max \|H111 − conditional\| | 0 |
| Max gain identity error | 0 |
| Max Shapley residual | 8.88e-16 |
| Max interaction residual | 0 |
| Max z reconstruction error | 0 |
| L9H0 / L9H7 P/V A↔B invariant | PASS (both 0) |
| NaN/Inf | none |
| Historical Phase-1/2/3 files | untouched |

## 10. Tokenization / template issues

None for GPT-2 Small: zero rejected names/places/objects. Variable-length templates pad with EOS; `end` never on pad.

## 11. Future full-run design (implemented, not executed)

- N=900 / seed = 30 examples × 30 strata; seeds 42,1,8,22 → 3600 prompts
- Bootstrap A: within-template stratified (preserve all 30 strata)
- Bootstrap B: template-cluster (resample 15 templates; keep BABA+ABBA paired)
- 10,000 draws; SHA256-stable seeds via `revision_stats.stable_sha256_seed`
- Per head × template × pattern reporting preserved (no premature averaging)

## 12. Key paths

- Module: `src/revision_standard_ioi.py`
- Stats: `src/revision_stats.py` (cluster + within-stratum bootstrap)
- Experiment: `experiments/revision_phase4_factorial.py`
- Modal: `modal_phase4.py`
- Smoke outputs: `results/revision_phase4/smoke/`
- Patch: `git_diff_phase4a.patch`
- Package: `phase4a_review_package.zip`

## STOP

Do **not** run N=900. Do **not** interpret N=60 science. Do **not** begin other models.
