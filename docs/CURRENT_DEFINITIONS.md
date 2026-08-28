# Current Method Definitions (Frozen Pre-Extended-FALI)

Terminology frozen for Phases 1–4. **Extended FALI is not defined here.**

---

## Contexts

| Symbol | Name | Definition |
|--------|------|------------|
| **A** | Intact context | Full model with primary name-movers **not** ablated |
| **B** | Primary-damaged context | Primary knockout active: **L9H6 + L9H9 + L10H0** zero-ablated |

Historical note: earlier work used B_old = L9H6+L9H9 only; revision Phases 1+ use the corrected primary above.

---

## Endpoint effects (per candidate head *h*)

Let IO−S = logit(IO) − logit(S) at the query/end position.

| Quantity | Definition |
|----------|------------|
| **intact_effect** | Effect of zeroing candidate *h* in context **A**: compares IO−S with vs without *h* ablated under intact primary context |
| **conditional_effect** | Effect of zeroing *h* in context **B**: compares IO−S with vs without *h* ablated under primary-damaged context |
| **conditional_gain** | `conditional_effect − intact_effect` |

Equivalent endpoint notation used in factorial cells:
- H000 ↔ intact baseline for decomposition
- H111 ↔ conditional endpoint
- gain identity: `(H111 − H000) = conditional_gain`

---

## Phase-3 factorial factors (P × V × C)

Intervention on candidate head full `hook_z` (all query positions).

| Factor | Symbol | Meaning |
|--------|--------|---------|
| **Pattern** | **P** | Candidate attention pattern (A vs B context cache) |
| **Value** | **V** | Candidate value vectors |
| **Context** | **C** | Surrounding model computation state |

### Important: what **C** is and is not

**C is NOT defined as purely “downstream sensitivity.”**

**C** is the **surrounding-network / context state** that differs between intact (A) and primary-damaged (B) runs **outside** the explicitly injected candidate P/V state in the factorial cell. It captures how the rest of the model’s computation differs when the primary movers are knocked out, while the candidate head receives a specified (P, V) injection.

Shapley components:
- `phi_pattern` (φ_P)
- `phi_value` (φ_V)
- `phi_context` (φ_C)

Plus main effects and interactions (`interaction_PV`, `interaction_PC`, `interaction_VC`, `interaction_PVC`) with exact reconstruction identities validated in Phase 3.

---

## Original FALI (descriptive)

**FALI** = Functional Attention Logit Attribution (descriptive, not causal by default).

Decomposes a change in the candidate head’s **target attention probability** into:
- **Target-score** contribution
- **Aggregate competing-score** contribution

Uses Shapley attribution over attention-score rows (target vs competing tokens). Valid as a descriptive decomposition of attention-score changes; **not** validated as a general cross-head predictor of functional recruitment (Phase 2 PIVOT).

---

## Phase-4 dataset naming

Experiment label: **canonical-template / public-implementation lexical distribution**

- 15 low-level BABA templates (match paper Appendix E)
- ABBA via Easy-Transformer first-clause swap
- Lexicon: 99 names, 8 places, 8 objects from public Easy-Transformer code (not a claim of exact paper lexical counts)

---

## Bootstrap procedures (Phase 4)

1. **Within-30-stratum bootstrap:** preserve all 30 template-pattern strata; resample examples within stratum.
2. **Template-cluster bootstrap:** resample 15 low-level templates as clusters; keep BABA+ABBA paired per sampled template.
3. **Hierarchical template-cluster (primary for broad-template claims):** sample template clusters; for each seed × template × pattern cell, **independently** resample prompts; equal average of four seed means. Seeds are replication cohorts, not resampled.

RNG: SHA256-stable seeds via `revision_stats.stable_sha256_seed`.

---

## Historical taxonomy vs measured functional role

Phase-1 **head_role** labels (backup, negative, extra_control) are **provenance taxonomy**, not a claim that a head has the same measured sign on every prompt distribution. Phase-4C reports measured roles separately (`functional_role_by_stratum.csv`).
