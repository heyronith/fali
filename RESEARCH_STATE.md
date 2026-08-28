# Research State — Pre-Extended-FALI Baseline

Factual snapshot as of repository tag `pre-extended-fali-baseline`.  
No new experiments; no Extended FALI implementation yet.

---

## A. Original FALI

**Research question:** Can changes in a backup attention head’s target-token probability be decomposed into target-score vs aggregate competing-score contributions (FALI), and does that decomposition predict functional backup recruitment?

**Method:** Target-score vs competing-score Shapley attribution of a target attention-probability change (descriptive decomposition on attention scores).

**Status:**
- Mathematically valid as a **descriptive** decomposition of attention-score changes.
- **Failed** as a general predictor/explanation of functional backup recruitment across heads (Phase 2B/3B **PIVOT** verdict: FALI anti-correlated with causal score hybrids at ~−0.35 Pearson across heads).

---

## B. Phase-1 — Corrected primary knockout & functional metrics

**Primary correction:** Canonical primary knockout = **L9H6 + L9H9 + L10H0** (replacing historical B_old = L9H6+L9H9 only).

**Core metrics (per candidate head *h*):**
- `intact_effect` = IO−S under intact context with *h* zeroed (vs clean baseline)
- `conditional_effect` = IO−S under primary-damaged context with *h* zeroed
- `conditional_gain` = `conditional_effect − intact_effect`

**Outcome:** Four-seed full runs (N=200/seed); corrected classification of 12 Phase-1 candidates; functional taxonomy documented under `results/revision_phase1/full/aggregate_corrected/`.

---

## C. Phase-2 — Causal score interventions & FALI validation

**Method:** Causal score hybrids (pattern / Q/K / combined), descriptive FALI, end-query causal totals on **12 heads** × 4 seeds × N=200.

**Verdict:** **PIVOT** — do not treat descriptive FALI as a reliable cross-head causal explanation; pursue structured causal interventions instead.

**Key outputs:** `results/revision_phase2/full/aggregate/` (head summaries, bootstrap, predictive CV, FALI↔causal correlations).

---

## D. Phase-3 — P×V×C factorial Shapley decomposition

**Factors:** P = candidate attention **pattern**, V = candidate **value**, C = surrounding **context** (intact A vs primary-damaged B).

**Intervention:** Full `hook_z` at all query positions for candidate head.

**Exact identities (validated):** H000 = intact_effect; H111 = conditional_effect; gain identity; 3-factor Shapley sum; interaction reconstruction; L9H0/L9H7 same-layer P/V invariants.

**Important heads (full N=200 × 4 seeds, 2-template IOI):**
- **L9H7:** φ_P ≈ 0, φ_V ≈ 0, φ_C > 0 (context recruitment)
- **L10H2:** φ_P > 0, gain > 0; φ_C often negative (pattern + opposing context)
- **L10H10:** φ_P > 0, φ_C < 0, cancellation / negative gain
- **L10H7:** large φ_P; conditional_effect sign unstable across template analyses

**Verdict:** **PARTIAL** (H1–H4 contrasts replicate; strict STRONG blocked by minor template/sign-flip issues).

---

## E. Phase-4 — Canonical templates & broad-template validation

**Dataset:** 15 Wang/Easy-Transformer BABA templates × ABBA = **30 strata**; public-implementation lexical inventory (99 names, 8 places, 8 objects). Not a claim of exact paper lexical reproduction.

**Full run:** N=900/seed (30 examples/stratum) × seeds {42, 1, 8, 22}; 12 heads; same Phase-3D factorial machinery.

**Bootstrap fix (Phase-4C):** Hierarchical template-cluster bootstrap corrected to resample prompts **independently per seed cell** while sharing template clusters across seeds. Means unchanged; CIs slightly wider; **no hypothesis status changes**.

**Locked verdict (corrected CIs):** **BROAD-TEMPLATE PARTIAL**
- H1–H3: strongly broadly supported
- H4 φ_P: strongly supported; H4 conditional_effect < 0: **NOT broadly supported** (measured aggregate conditional_effect positive on canonical IOI)

---

## F. Current research direction (not yet implemented)

**Candidate direction:** Extended / Causal FALI — explain changes in a candidate attention head's causal usefulness through structured interventions over attention-score computation, value content, and surrounding-network context, with a nested target-versus-competitor decomposition inside the attention-score channel.

**This is a candidate direction only.** No formal definition, proof, or implementation in this baseline.

---

## G. Unresolved gates (before Extended FALI claims)

1. Formal mathematical definition and proof obligations
2. Synthetic ground-truth validation
3. Ablation / intervention robustness (not yet run)
4. Cross-task and cross-model generalization

Do not claim success on Extended FALI until these gates are addressed.
