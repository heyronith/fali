# Final Scientific Audit — Machine Learning Manuscript

**Audit date:** 2026-08-29  
**Scope:** claim→evidence traceability; claim→citation/novelty; internal consistency; adversarial limitations.  
**Experimental phase:** CLOSED. No new experiments authorized or required by this audit.

**Verdict**

**PASS**

The central scientific story is supported and internally coherent:

- GPT-2 IOI L10H2 is positively recruited and has FALI target-positive/rival-negative score contributions.
- GPT-2 L10H10 shows strong target-attention adaptation and the same fine-score signs, but negative broad-canonical conditional gain; attention adaptation is therefore not equivalent to positive recruitment.
- The preregistered EF-4 fine-score gate passes.
- EF-5 matched-mean and matched-resample fine-channel analyses are ROBUST under their preregistered channel criteria, while the overall EF-5 gate FAILS because the zero-damage reference perturbation criterion is not satisfied.
- Pythia-160M induction L8H2 is positively recruited after L5H0 perturbation, but the preregistered GPT-2 fine-score signs reverse: phi_T<0 and phi_R>0.
- G0–G2 confirm and G3–G4 do not; EF-6 therefore FAILS as a mechanism-transfer hypothesis.
- The cross-setting conclusion is bounded to the two studied settings: positive conditional recruitment occurs in both studied cases, while the preregistered fine-score mechanism does not transfer.

No audit finding requires a new model, task, seed, ablation, or GPU run.

---

## 1. Claim → evidence audit

### Method / validation

Primary source:

- `results/extended_fali/ef1_synthetic/validation_report.json`

Verified:

- `overall_pass = true`.
- Five single-channel constructions recover the intended T/R/O/V/C channel; maximum intended-channel discrepancy is `1.1102230246251565e-16`.
- Pure T×C construction has DeltaU≈2 and an equal standard-Owen split, phi_T≈1 and phi_C≈1.
- Random two-layer TransformerLens test recovers score→pattern, pattern/value→z, independent endpoints, full-z injection isolation, zero live Q/V/QV leakage, and quotient consistency at recorded precision.
- Direct endpoint quantities are intact effect `0.4550457001`, conditional effect `-0.0856835023`, conditional gain `-0.5407292023`.

**Status:** GREEN. Main-text validation claims are traceable.

### GPT-2 canonical fine decomposition

Primary sources:

- `results/extended_fali/ef4_canonical/aggregate/exact_means.csv`
- `results/extended_fali/ef4_canonical/aggregate/hierarchical_bootstrap_ci.csv`
- `results/extended_fali/ef4_canonical/aggregate/fine_score_gate.json`

Verified anchors:

| Head | Quantity | Point estimate | 95% CI |
|---|---:|---:|---:|
| L10H2 | conditional gain | 0.298451 | [0.262940, 0.335184] |
| L10H2 | phi_T | 0.782806 | [0.721856, 0.838338] |
| L10H2 | phi_R | -0.341419 | [-0.371673, -0.310909] |
| L10H2 | Phi_SCORE | 0.445252 | [0.408802, 0.477107] |
| L10H2 | phi_C | -0.146451 | [-0.164685, -0.128324] |
| L10H10 | conditional gain | -0.263442 | [-0.301914, -0.224875] |
| L10H10 | phi_T | 0.328524 | [0.300205, 0.355043] |
| L10H10 | phi_R | -0.125907 | [-0.138610, -0.113054] |
| L10H10 | Phi_SCORE | 0.206251 | [0.185089, 0.225661] |
| L10H10 | phi_C | -0.471249 | [-0.514272, -0.427467] |

EF-4 fine-score gate: **PASS**.

### GPT-2 broad-canonical functional effects

The conditional-effect quantities are not in EF-4 `exact_means.csv`; they come from the corrected historical Phase-4C broad-template analysis. The evidence map must therefore keep these paths explicit:

- `results/revision_phase4/full/aggregate/L10H2_broad_template_report.md`
- `results/revision_phase4/full/aggregate/L10H10_broad_template_report.md`

Verified:

- L10H2 conditional effect `0.382239`, hierarchical CI `[0.340647, 0.423865]`.
- L10H10 conditional effect `-0.078890`, hierarchical CI `[-0.111127, -0.045726]`.

This distinction matters because positive functional recruitment requires both positive conditional effect and positive conditional gain.

**Evidence-ledger maintenance patch:** add the Phase-4C broad-template reports to the evidence paths for claims F2/G1 (L10H2 positive functional recruitment) and F3 (L10H10 broad-canonical negative conditional effect), rather than implying EF-4 `exact_means.csv` alone contains both quantities.

### EF-5 intervention robustness

Primary sources:

- `results/extended_fali/ef5_ablation_robustness/aggregate/ablation_gate.json`
- `results/extended_fali/ef5_ablation_robustness/aggregate/channel_robustness.json`

Verified:

- `EF5_ABLATION_GATE = FAIL`.
- `matched_mean_status = ROBUST`.
- `matched_resample_status = ROBUST`.
- Failure reason: zero damage-drop hierarchical CI is not entirely >0; the preregistered reference-perturbation assumption fails.
- All four frozen fine channels retain their predicted signs under both tested matched alternatives; all 30 strata have the frozen sign in each channel.

**Status:** GREEN only with bounded wording. Never convert the channel-level ROBUST results into an overall EF-5 pass or arbitrary-ablation robustness claim.

### Pythia induction transfer

Primary sources:

- `results/extended_fali/ef6_pythia_induction/aggregate/exact_means.csv`
- `results/extended_fali/ef6_pythia_induction/aggregate/ef6_gate.json`
- `results/extended_fali/ef6_pythia_induction/aggregate/hypothesis_results.json`

Verified:

| Quantity | Point estimate | 95% CI |
|---|---:|---:|
| induction advantage | 14.017091 | [12.280371, 15.939234] |
| primary drop | 1.145992 | [0.886752, 1.538771] |
| conditional effect | 0.621197 | [0.455912, 0.860486] |
| conditional gain | 0.485653 | [0.326486, 0.721815] |
| phi_T | -0.143737 | [-0.247435, -0.084085] |
| phi_R | +0.117800 | [0.058225, 0.219125] |
| phi_O | -0.000270 | [-0.000847, 0.000256] |
| phi_V | -0.002099 | [-0.002762, -0.001557] |
| phi_C | +0.513958 | [0.354460, 0.752149] |

Gate state:

- G0 CONFIRMED
- G1 CONFIRMED
- G2 CONFIRMED
- G3 NOT_CONFIRMED
- G4 NOT_CONFIRMED
- `EF6_GATE = FAIL`

**Status:** GREEN. The failed transfer hypothesis must remain a negative result, not be reframed as confirmation of a universal opposite mechanism.

---

## 2. Claim → citation / novelty audit

### Allowed positioning

The manuscript correctly treats FALI as downstream of:

1. self-repair / backup-component work;
2. conditional component importance (especially CoAx);
3. interaction-aware causal attribution (especially *The Curse of Multiple Mediators*);
4. activation/path patching;
5. established QK/OV and recent causal routing-versus-value decomposition;
6. standard Owen/Shapley attribution.

The contribution is the **specific counterfactual estimand and architecture-aligned intervention game**, not discovery of self-repair, conditionality, routing/value separation, or Owen mathematics.

### Qualified novelty claim

The only first/precedence claim allowed is:

> To our knowledge, FALI is the first framework to decompose the intervention-induced change in an attention head's output-grounded causal usefulness into architecture-aligned score, value, and surrounding-context contributions while resolving the score contribution into target, rival, and other-score channels.

Keep `To our knowledge`. Do not upgrade to an unconditional first claim.

**Mechanical manuscript patch required:** attach the closest-work citation cluster directly to this sentence:

```tex
\citep{gong2026coax,vaidyanathan2026curse,wang2026fewshot,zhu2026lexical}
```

The following sentence can continue to distinguish the claim from established routing/content separation and cite QK/OV / attention-causal work.

### Final adversarial search outcome

Fresh search through 2026-08-29 found no direct prior framework matching the full combination:

- cross-state change in one selected attention head's output-grounded causal usefulness;
- explicit SCORE/VALUE/CONTEXT decomposition;
- target/rival/other-score resolution inside SCORE;
- surrounding-context factor;
- structured interaction allocation with coarse/fine reconciliation.

Two August-2026 weight-space-ablation preprints surfaced as additional neighboring work on conditional collapse / cross-layer interaction under ablation. They strengthen the case that conditional interaction is an active neighboring area, but they do not appear to implement the FALI estimand or T/R/O/V/C game. They do **not** justify broadening the novelty claim.

**Novelty status:** YELLOW/GREEN only in the qualified form above. Re-run a final title/abstract/keyword search immediately before actual submission if submission is delayed materially.

### Bibliography metadata note

`references.bib` is broadly consistent with the cited primary sources. One minor metadata cleanup should be checked mechanically against the PMLR MIB record: the author currently listed as `Fiotto-Kaufman, Jaden` appears in the primary proceedings metadata as `Jaden Fried Fiotto-Kaufman`. Correct the BibTeX name if the proceedings record confirms that exact form; this is bibliographic metadata only and does not affect scientific claims.

---

## 3. Internal-consistency audit

### Terminology / estimands

Consistent and approved:

- FALI = Factorized Attention-Level Intervention.
- `conditional gain = E_B(h)-E_A(h)`.
- positive functional recruitment requires `E_B(h)>0` and conditional gain >0.
- T/R/O are pre-softmax score-coordinate factors.
- V is candidate value content.
- C is surrounding-network/context contribution, not a uniquely localized external pathway.
- Standard Owen partition: SCORE={T,R,O}, VALUE={V}, CONTEXT={C}; partition fixed, block order averaged.
- 32 cells; 36 admissible block-contiguous permutations.

### Required metric wording correction

`main.tex` currently describes the second example metric as an "induction-specific logit score." The Pythia experiment uses **target-token log probability**.

Mechanical replacement:

```text
OLD:
The task metric can differ across applications---for example, an IO-minus-subject logit difference in IOI or an induction-specific logit score---but it is fixed within an analysis.

NEW:
The task metric can differ across applications---for example, an IO-minus-subject logit difference in IOI or the target-token log probability in induction---but it is fixed within an analysis.
```

### Pythia heading scope

Resolved during audit:

- Old: `Recruitment generalizes but the fine mechanism reverses`
- New: `Positive recruitment recurs while the fine mechanism reverses`

This avoids an unqualified generalization in a heading while preserving the empirical result.

### Owen wording in contribution sheet

Resolved during audit:

- replaced `fixed coalition structure` with `fixed coalition partition` to avoid any accidental echo of the superseded fixed-block-order formulation.

### Figures / captions

Approved and internally consistent:

- Fig. 1: selected conditional change → T/R/O/V/C explanation.
- Fig. 2: valid score partition and 32-cell game; `(q*,q*)` correctly belongs to R in the illustrative case.
- Fig. 3: implementation/ground-truth validation only.
- Fig. 4: L10H2 recruitment versus L10H10 attention adaptation without recruitment.
- Fig. 5: both studied settings have positive conditional gain for the selected recruited heads, with opposite T/R signs and EF-6 transfer FAIL.

No figure should be reinterpreted as evidence for a unique natural causal pathway.

---

## 4. Adversarial reviewer / limitations audit

The manuscript now explicitly covers the main scientific threats:

### Threat model: false mechanistic inference

Potential causes:

- off-manifold hybrid states;
- post-hoc candidate/target selection;
- intervention-family dependence;
- coalition/grouping dependence;
- treating C as a localized pathway;
- confusing attention change with causal usefulness;
- selectively reporting successful fine-channel robustness while hiding the EF-5 experiment-level failure;
- reframing failed Pythia transfer as a successful universal mechanism claim.

Mitigations present in manuscript/project:

- explicit counterfactual semantics;
- selected head/target scope stated;
- preregistered/frozen confirmation hypotheses and transfer gates;
- EF-5 overall failure reported;
- EF-6 failed transfer reported;
- C explicitly described as coarse surrounding context;
- validation separates implementation correctness from natural-pathway claims.

### Scaling risk

The five-factor game has 32 cells per example/candidate. This is practical for hypothesis-driven selected-head analysis but is not presented as an exhaustive whole-model scanning method. Scaling to many heads/edges would multiply intervention cost and researcher degrees of freedom.

### Gaming potential

The main gaming risk is selection after seeing attribution results: choosing favorable heads, target edges, templates, or intervention variants. The current confirmation stages reduce this by freezing the relevant panels, directions, and transfer pair before confirmation. Future work using FALI should preserve that separation between discovery and confirmation.

### New attack surface

No meaningful external security/capability attack surface is introduced beyond ordinary local mechanistic-intervention analysis. The primary risk is epistemic: producing persuasive but intervention-contingent mechanistic narratives.

### Scope limits that must remain

- two substantive settings only;
- attention-head-specific algebra;
- target edge/set must be specified;
- C remains coarse;
- mixed cells may be off-manifold;
- attributions depend on intervention and partition choices;
- no transformer-wide universality claim;
- no arbitrary-ablation robustness claim.

---

## Mechanical patches before final audit PASS

1. **`main.tex`: metric description** — replace `induction-specific logit score` with `the target-token log probability in induction`.
2. **`main.tex`: novelty sentence** — attach `\citep{gong2026coax,vaidyanathan2026curse,wang2026fewshot,zhu2026lexical}` directly to the qualified first-framework sentence.
3. **`EVIDENCE_LEDGER.md`: functional-effect provenance** — add Phase-4C L10H2/L10H10 broad-template report paths for the canonical conditional-effect claims / positive-recruitment evidence.
4. **`references.bib`: MIB metadata check** — if confirmed against PMLR, expand `Fiotto-Kaufman, Jaden` to the proceedings form `Fiotto-Kaufman, Jaden Fried`.
5. Recompile `main.tex` and `contribution_information_sheet.tex`; require 0 undefined citations/references and no substantive page/layout regression. 20 pages is ideal; 21–22 remains acceptable after the added limitations text.
6. Run `python -m pytest tests/ -q` and `render_all.py`; empirical outputs/gates must remain unchanged.

After these mechanical patches and a clean compile/test pass, the scientific audit status becomes **PASS**. Remaining work is submission metadata + supplement completion, not further experimentation or scientific-story changes.

## Completion note (mechanical patches applied)

- **Patch commit SHA:** 29a28094e5e5692235d55085db14dccc1f2955fa
- **Compile status:** PASS (`main.tex`, `contribution_information_sheet.tex`)
- **Final page count:** 21 pages (`main.pdf`; +1 vs. pre-patch 20-page recompile due to expanded limitations text and inline novelty citations; within acceptable 21–22 band)
- **Tests status:** `pytest tests/ -q` → 205 passed
- **Empirical outputs:** `render_all.py` → VALIDATION PASS; EF-1R PASS, EF-4 PASS, EF-5 overall FAIL, EF-6 FAIL unchanged; all point estimates/CIs unchanged
