# Figure Plan — Machine Learning Manuscript

Status: **FROZEN MAIN-FIGURE PLAN**
Target: **4 main figures**. A fifth robustness figure is optional and should remain supplementary unless it materially improves reviewer comprehension.

All figures must be reproducible from committed code/results or deterministic vector source. No decorative/generative-AI artwork.

## Figure 1 — What FALI explains

**Purpose:** establish the estimand and distinguish detection from explanation.

**Layout:** three panels.

- **(a) Conditional usefulness:** intact state A vs perturbed state B; show candidate head usefulness measured by candidate-present minus candidate-zero output in each context; visually define conditional gain as the cross-state change.
- **(b) FALI factorization:** candidate attention computation split into T target score(s), R rival scores, O other valid scores, V value content, and C surrounding-network/context.
- **(c) Attribution structure:** SCORE={T,R,O}, VALUE={V}, CONTEXT={C}; show coarse/fine reconciliation and output-grounded utility.

**Data source:** none; deterministic schematic derived directly from the formal specification.

**Implementation:** vector/TikZ or reproducible SVG/PDF source. No neural-network clip art.

**Caption claim:** FALI decomposes a selected head's cross-state change in output-grounded causal usefulness under an explicit counterfactual game; it does not perform circuit discovery.

**Threat to avoid:** implying that T/R/O/V/C are unique natural pathways or that hybrid states occur naturally.

---

## Figure 2 — Ground-truth and implementation validation

**Purpose:** demonstrate that the method identifies known factors before empirical interpretation.

**Suggested layout:** two compact panels.

- **(a) Single-channel recovery:** five rows/cases (T-only, R-only, O-only, V-only, C-only); plot true total change and recovered Owen attribution. Untouched channels should be zero within preregistered numerical tolerance.
- **(b) Implementation invariants:** concise validation strip/table: endpoint identity, score→pattern reconstruction, pattern/value→z reconstruction, no-live-candidate leakage, untouched-head scope, quotient bridge.

**Primary source:** `results/extended_fali/ef1_synthetic/validation_report.json`.

**Supporting source/code:**
- `src/extended_fali_math.py`
- `src/extended_fali_interventions.py`
- `docs/EXTENDED_FALI_FORMAL_SPEC.md`

**Main-text claim allowed:** the mathematical/intervention pipeline passed controlled factor-identification and implementation-invariant tests.

**Do not claim:** synthetic validation proves real-model mechanistic truth.

---

## Figure 3 — GPT-2: attention adaptation and causal recruitment are different

**Purpose:** establish the corrected functional problem and canonical GPT-2 mechanism.

**Suggested layout:** three panels.

- **(a) Functional contrast:** L10H2 vs L10H10. Show corrected conditional gain alongside change in target attention. L10H2 is genuinely recruited; L10H10 increases target attention but has negative conditional gain.
- **(b) Canonical FALI:** point estimates + hierarchical 95% CIs for L10H2 and L10H10 fine channels T and R; optionally O as a muted descriptive marker.
- **(c) Coarse context (small inset only if legible):** SCORE/V/C totals showing why positive score adaptation need not imply positive overall gain.

**Primary sources:**
- `results/extended_fali/ef4_canonical/aggregate/exact_means.csv`
- `results/extended_fali/ef4_canonical/aggregate/hierarchical_bootstrap_ci.csv`
- `results/extended_fali/ef4_canonical/aggregate/fine_score_gate.json`

**Functional-baseline source:**
- `results/revision_phase1/full/aggregate_corrected/functional_classification.csv`
- `results/revision_phase1/full/aggregate_corrected/canonical_backup_summary.csv`

**Locked numerical anchors from EF-4:**
- L10H2 conditional gain ≈ 0.298; 95% CI [0.263, 0.335]
- L10H2 phi_T ≈ 0.783; CI [0.722, 0.838]
- L10H2 phi_R ≈ -0.341; CI [-0.372, -0.311]
- L10H10 conditional gain ≈ -0.263; CI [-0.302, -0.225]
- L10H10 delta target attention ≈ +0.310; CI [0.285, 0.335]
- L10H10 phi_T ≈ +0.329; CI [0.300, 0.355]
- L10H10 phi_R ≈ -0.126; CI [-0.139, -0.113]

**Caption claim:** canonical GPT-2 results separate visible attention adaptation from functional recruitment and identify positive target / negative rival score contributions under the FALI game.

---

## Figure 4 — Principal empirical result: same recruitment label, opposite fine mechanism

**Purpose:** visual centerpiece of the paper.

**Layout:** side-by-side GPT-2 IOI vs Pythia-160M induction, with aligned axes for T and R signs. Include conditional gain above each setting to show that the high-level functional phenomenon occurs in both.

**GPT-2 source:**
- `results/extended_fali/ef4_canonical/aggregate/exact_means.csv`
- `results/extended_fali/ef4_canonical/aggregate/hierarchical_bootstrap_ci.csv`

Preferred GPT-2 exemplar: L10H2, because it is a genuine recruited backup under the corrected functional definition.

**Pythia source:**
- `results/extended_fali/ef6_pythia_induction/aggregate/exact_means.csv`
- `results/extended_fali/ef6_pythia_induction/aggregate/bootstrap_ci.csv`
- `results/extended_fali/ef6_pythia_induction/aggregate/ef6_gate.json`

**Locked Pythia anchors:**
- conditional effect ≈ 0.621; CI [0.456, 0.860]
- conditional gain ≈ 0.486; CI [0.326, 0.722]
- phi_T ≈ -0.144; CI [-0.247, -0.084]
- phi_R ≈ +0.118; CI [0.058, 0.219]

**Visual claim:** both studied settings exhibit positive conditional recruitment, while the target/rival signs are opposite.

**Caption wording:** "Conditional recruitment occurs in both studied settings, but the fine target/rival score contributions differ in sign. The Pythia experiment therefore rejects the preregistered GPT-2 sign-transfer hypothesis."

**Do not write:** "FALI proves there is no universal repair mechanism."

---

## Optional Figure 5 / Supplementary Figure — Intervention-scheme robustness

**Default location:** Supplementary Information.

**Purpose:** show matched alternative ablation schemes without overstating EF-5.

**Sources:**
- `results/extended_fali/ef5_ablation_robustness/aggregate/ablation_gate.json`
- `results/extended_fali/ef5_ablation_robustness/aggregate/channel_robustness.json`
- `results/extended_fali/ef5_ablation_robustness/aggregate/exact_means.csv`
- `results/extended_fali/ef5_ablation_robustness/aggregate/hierarchical_bootstrap_ci.csv`

**Required annotation:** matched_mean = ROBUST; matched_resample = ROBUST; overall EF5_ABLATION_GATE = FAIL because the zero-damage reference perturbation criterion failed.

No figure may summarize this as simply "robust across ablations."

---

## Visual standards

- Vector PDF/EPS preferred for diagrams and line/bar/interval plots.
- Plot scripts must read source CSV/JSON; no manually entered bar heights.
- Show uncertainty intervals for inferential empirical quantities.
- Use shapes/line styles in addition to color so plots remain interpretable in grayscale and for color-vision deficiencies.
- Avoid legends when direct labels are clearer.
- No chart title embedded inside the figure; title-level information belongs in the caption.
- Consistent 8–12 pt final-size lettering.
- Figure filenames for submission should follow Springer style (`Fig1.pdf`, `Fig2.pdf`, ...).
