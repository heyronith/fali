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

---

## D. EF-3 — Two-template full scientific analysis (frozen before execution)

**Purpose:** First scientific Extended-FALI experiment. Decompose fine-grained score
mechanisms (T/R/O/V/C) under the preregistered two-template mixed distribution.

**Scope:**
- N = 200 per seed; seeds = 42, 1, 8, 22 (800 prompts total)
- template_style = `"mixed"` (100 A + 100 B per seed)
- All 12 Phase-1 candidates
- Per-example Owen attribution before aggregation

**Preregistration statements (frozen before EF-3 execution):**
- No EF-2 N=8 attribution values were used to modify hypotheses.
- H1–H4 remain exactly those frozen before EF-2 (see below).
- Canonical-template data remain untouched confirmation data.

**EF-3 is:**
- Confirmatory for H1–H3 (components as preregistered)
- Exploratory for R/O unless discovery rule passes
- Discovery-only for any NEW mechanism promoted to EF-4

### Primary estimand

Balanced mean over 4 seeds × 2 templates × 100 prompts per head.

Seeds are **prompt-generation cohorts**, not model replicas.

### Primary bootstrap

10,000 deterministic SHA256-seeded draws. Within each draw, resample 100 prompts
with replacement inside each seed×template cell; equal A/B within seed; equal
across four seeds.

### Confirmatory hypotheses (unchanged from pre-EF-2 freeze)

**H1 — L9H7 (L9H0 control):** |φ_T|,|φ_R|,|φ_O|,|φ_V| ≤ 1e−6; φ_C = conditional_gain
within 1e−6 per example (800 prompts).

**H2 — L10H2:** conditional_gain > 0, Phi_SCORE > 0, φ_T > 0. Support requires 95%
bootstrap CI entirely in direction AND all 4 seed means in direction. φ_R, φ_O exploratory.

**H3 — L10H10:** Phi_SCORE > 0, φ_C < 0, conditional_gain < 0, φ_T > 0. Same support rule.
φ_R, φ_O exploratory.

**H4 — Descriptive contrast:** L10H2 vs L10H10 side-by-side (attention + causal channels).
Not a separate null test.

### R/O discovery rule (L10H2, L10H10 only)

Promote to EF-4 candidate only if ALL: CI excludes zero; 4/4 seed means same sign;
|mean channel| ≥ 10% × |mean Phi_SCORE|.

### GO/STOP to canonical confirmation

GO only if: (A) all implementation validation passes; (B) L9H7 architectural case holds;
(C) ≥1 stable T/R/O channel on L10H2 or L10H10 passes discovery rule.

Otherwise STOP before canonical templates.

### Implementation validation (every seed)

Same thresholds as EF-2: score→pattern ≤2e−6; z ≤1e−5; endpoints ≤1e−4; Owen ≤1e−8;
quotient ≤1e−8; L9 score/V ≤1e−6; historical bridge ≤1e−4.

### Failure policy

Any seed validation failure → STOP scientific interpretation.

No threshold changes post hoc. No alternate seeds.

### Deliverables

```
results/extended_fali/ef3_two_template/seed_<seed>/
results/extended_fali/ef3_two_template/aggregate/
```

Raw cell files indexed on branch `audit/extended-fali-ef3-raw` (private).
