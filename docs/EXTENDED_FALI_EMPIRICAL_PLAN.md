# Extended FALI — Empirical Plan (frozen before EF-2)

This document distinguishes implementation validation from future scientific analyses.
It is frozen before any GPT-2 execution in Phase EF-2.

---

## A. EF-2 — GPT-2 small IOI implementation smoke (N=8, seed=42)

**Purpose:** First pretrained-model validation that the Extended-FALI score → pattern →
value → z → causal-utility pipeline matches historical Phase-3C P×V×C semantics on
GPT-2 IOI.

**Scope:**
- N = 8, seed = 42, template_style = `"mixed"`
- Same dataset cohort as `results/revision_phase3c/smoke/`
- All 12 Phase-1 candidate heads
- 32 Extended-FALI cells per head × example
- Independent bridge to saved `factorial_raw.csv` H-cells and Shapley attributions

**EF-2 N=8 outputs are validation data only and will not constitute evidence for or
against scientific hypotheses.**

No hypothesis testing, bootstrapping, or sign interpretation in EF-2 reports.

---

## B. Future two-template discovery analysis (not yet run)

**Purpose:** Exploratory attribution on the two-template mixed cohort at larger N
(e.g. N=200 per seed) to identify candidate mechanisms.

**Status:** Planned after EF-2 smoke passes. Requires separate preregistration amendment
before execution.

**Exploratory factors:** φ_R and φ_O are exploratory in the discovery stage for
L10H2 and L10H10. No directional T/R/O hypothesis is frozen for L10H7.

---

## C. Future canonical-template confirmation analysis (not yet run)

**Purpose:** Independent confirmation on canonical IOI templates (Phase-4 style).

**Status:** Planned after discovery cohort. Any R/O mechanism discovered in (B) may
become confirmatory only if frozen **after** the discovery cohort and tested on the
independent canonical-template cohort.

---

## Frozen hypotheses (from Phase 1–4 evidence only)

These guide **future** interpretation; EF-2 does not test them statistically.

### H1 — Same-layer context-only case (L9H7)

Under canonical primary knockout and Extended-FALI intervention family:

- Score block contribution ≈ 0
- φ_T ≈ 0, φ_R ≈ 0, φ_O ≈ 0, φ_V ≈ 0
- Conditional usefulness change attributable to C

**Architectural control:** L9H0 (same layer, different head).

### H2 — L10H2

- conditional_gain > 0
- SCORE block / historical pattern contribution > 0
- φ_T predicted > 0
- φ_R and φ_O **exploratory** in two-template discovery stage

### H3 — L10H10

- SCORE block / historical pattern contribution > 0
- φ_C < 0
- conditional_gain < 0 on previously studied distributions
- φ_T predicted > 0
- φ_R and φ_O **exploratory** initially

### H4 — Attention / function distinction

L10H2 and L10H10 may show similar target-directed attention adaptation while
differing in causal usefulness because Extended FALI includes score/value/context
channels.

### Non-hypothesis

No directional T/R/O hypothesis is frozen for **L10H7**.

---

## EF-2 validation criteria (reference)

See `docs/EXTENDED_FALI_PREREGISTRATION.md` EF-1/EF-1R sections for synthetic
thresholds. EF-2 GPT-2 thresholds are defined in the research-lead EF-2 instruction
(implementation smoke commit message and `validation_report.json`).

**Failure policy:** Any primary EF-2 validation failure → STOP. Do not loosen
thresholds, change seed, or exclude failing heads.

---

## Deliverables (EF-2 smoke)

```
results/extended_fali/ef2_gpt2_smoke/
├── run_config.json
├── environment.json
├── dataset_metadata.csv
├── dataset_alignment_report.md
├── cell_raw.csv
├── attribution_raw.csv
├── head_summary.csv
├── validation_report.json
└── historical_bridge_comparison.csv
```

Historical Phase 1–4 files must remain unchanged.
