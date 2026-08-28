# Research State — Pre-Extended-FALI Baseline

Factual snapshot as of repository tag `pre-extended-fali-audited` (successor audit to unchanged `pre-extended-fali-baseline`).

**EF-0 update:** Extended FALI mathematical specification implemented and synthetically algebra-tested; no model-level validation yet.

No GPT-2 / Modal empirical runs for Extended FALI.

---

## A. Original FALI

**Research question:** Can changes in a backup attention head’s target-token probability be decomposed into target-score vs aggregate competing-score contributions (path-stable FALI analysis), and does that decomposition predict functional backup recruitment?

**Historical intervention name:** **FALI** = **Frozen Attention Logit Intervention** (pre-softmax target-vs-nontarget counterfactual in the original manuscript).

**Later descriptive analysis:** Two-player Shapley decomposition of target attention probability into target-score vs aggregate competing-score contributions on attention scores.

**Status:**
- Mathematically valid as a **descriptive** decomposition of attention-score changes.
- **Failed** as a general predictor/explanation of functional backup recruitment across heads (Phase 2B/3B **PIVOT** verdict: FALI anti-correlated with causal score hybrids at ~−0.35 Pearson across heads).

---

## B. Phase-1 — Corrected primary knockout & functional metrics

**Primary correction:** Canonical primary knockout = **L9H6 + L9H9 + L10H0** (replacing historical **B_old = L9H9 + L10H0**).

**Endpoint notation (per candidate head *h*; IO−S = logit(IO) − logit(S) at end position):**

| Symbol | Meaning |
|--------|---------|
| **A_h** | Clean/intact IO−S with candidate *h* present |
| **D_h** | Intact IO−S with candidate *h* zeroed |
| **B_h** | Primary-damaged IO−S with candidate *h* present |
| **C_h** | Primary-damaged IO−S with candidate *h* zeroed |

**Derived effects:**

- `intact_effect` = **A_h − D_h**
- `conditional_effect` = **B_h − C_h**
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

**Exact identities (validated):** H000 = intact baseline for decomposition; H111 = conditional endpoint; gain identity; 3-factor Shapley sum; interaction reconstruction; L9H0/L9H7 same-layer P/V invariants.

With endpoint notation above: H000 ↔ comparison involving **A_h** vs **D_h**; H111 ↔ comparison involving **B_h** vs **C_h**; `(H111 − H000) = conditional_gain`.

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

## F. Current research direction (EF-0 formalization)

**Extended / Causal FALI** — hierarchical five-player Owen attribution of candidate-head
causal usefulness Δ_U over score blocks (T,R,O), value (V), and context (C).

**EF-0 status:** Formal spec + pure game mathematics in `src/extended_fali_math.py`;
synthetic algebra tests pass. See `docs/EXTENDED_FALI_FORMAL_SPEC.md` and
`docs/EXTENDED_FALI_PREREGISTRATION.md`.

**Not yet done:** TransformerLens implementation, synthetic transformer ground-truth,
GPT-2 IOI runs, empirical head hypotheses.

---

## G. Unresolved gates (before Extended FALI claims)

1. Formal mathematical definition and proof obligations
2. Synthetic ground-truth validation
3. Ablation / intervention robustness (not yet run)
4. Cross-task and cross-model generalization

Do not claim success on Extended FALI until these gates are addressed.
