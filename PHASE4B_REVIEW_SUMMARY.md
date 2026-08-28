# PHASE4B Review Summary — Full Broad-Template Validation

**Status:** All four N=900 Modal runs PASS; aggregation complete  
**Date:** 2026-08-14  
**Experiment:** canonical-template / public-implementation lexical distribution  
**Verdict:** **BROAD-TEMPLATE PARTIAL**

Hypotheses were **not** redefined after observing results. Manuscript not rewritten.

## Lexical / template provenance

- 15 low-level BABA templates match paper Appendix E.
- Public Easy-Transformer lexical inventory used: **99 names, 8 places, 8 objects**.
- Paper text describes ~100 names and ~20 total common place/object words.
- This run does **not** claim exact reproduction of every lexical detail of the paper distribution.

## Full run

| Seed | N | Strata balance | Acceptance |
|------|--:|----------------|------------|
| 42 | 900 | 30×30 | PASS |
| 1 | 900 | 30×30 | PASS |
| 8 | 900 | 30×30 | PASS |
| 22 | 900 | 30×30 | PASS |

Total prompt instances: **3600**. All 12 heads. Modal NVIDIA L4 FP32.

Across seeds: **120 examples per template-pattern stratum**.

## Primary hypothesis evaluation (all-30 hierarchical template-cluster CI)

| Family | Quantity | Status |
|--------|----------|--------|
| H1 L9H7 | φ_P ≈ 0 | ARCHITECTURAL_INVARIANT_PASS |
| H1 L9H7 | φ_V ≈ 0 | ARCHITECTURAL_INVARIANT_PASS |
| H1 L9H7 | φ_C > 0 | STRONGLY_BROADLY_SUPPORTED |
| H2 L10H2 | gain > 0 | STRONGLY_BROADLY_SUPPORTED |
| H2 L10H2 | φ_P > 0 | STRONGLY_BROADLY_SUPPORTED |
| H3 L10H10 | φ_P > 0 | STRONGLY_BROADLY_SUPPORTED |
| H3 L10H10 | φ_C < 0 | STRONGLY_BROADLY_SUPPORTED |
| H3 L10H10 | gain < 0 | STRONGLY_BROADLY_SUPPORTED |
| H4 L10H7 | φ_P > 0 | STRONGLY_BROADLY_SUPPORTED |
| H4 L10H7 | conditional_effect < 0 | **NOT_BROADLY_SUPPORTED** (mean ≈ +0.30) |

→ H1–H3 families strongly broadly supported; H4 fails net-suppressive arm → **BROAD-TEMPLATE PARTIAL**.

## Secondary confirmatory

| ID | Result |
|----|--------|
| S1 L10H10 interaction_PC < 0 | STRONGLY_BROADLY_SUPPORTED |
| S2 L10H2 φ_context < 0 | STRONGLY_BROADLY_SUPPORTED |
| L10H1 φ_context > 0 (suggestive) | NOT_BROADLY_SUPPORTED (mean negative) |

## Non-overlap (29 strata, exclude exact-overlap T00_BABA)

Same qualitative status as all-30 for all primary/secondary rows above.

## BABA vs ABBA

Pattern-type aggregates agree in sign for primary heads on key metrics; paired template diffs flag some asymmetries (see `baba_abba_summary.csv`). No averaging-away of reversals.

## Value contribution

|φ_value| remains much smaller than |φ_pattern| and |φ_context| for backup heads (GPT-2 Small IOI only).

## Unexpected finding

Under the **2-template** Phase-3D design, L10H7 had negative `conditional_effect`. Under the **canonical 30-strata** distribution, L10H7 retains large positive φ_P but **positive** conditional_effect in all 4 seeds — so H4’s “adaptive-but-suppressive” net function does **not** broadly replicate, while its pattern adaptation does.

## Artifacts

- Aggregate: `results/revision_phase4/full/aggregate/`
- Raw factorial CSVs: see `raw_factorial_index.json` (SHA256 + schema; omitted from ZIP for size)
- Package: `phase4b_review_package.zip`

## STOP

No other models. No alternate ablations. No manuscript rewrite.
