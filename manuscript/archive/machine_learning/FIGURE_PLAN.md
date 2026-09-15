# Figure Plan — Machine Learning Manuscript

Status: **REVISED AND FROZEN MAIN-FIGURE PLAN**
Target: **5 main figures**. EF-5 robustness remains supplementary unless page/layout review later demonstrates a clear need to promote it.

Revision rationale: the manuscript introduces a new mechanism-resolution method, so the original four-figure plan was expanded to separate (i) the scientific question from (ii) the intervention mechanics. This avoids overloading a single conceptual figure and improves reviewer comprehension.

All figures must be reproducible from committed code/results or deterministic vector source. No decorative/generative-AI artwork.

## Figure 1 — From conditional importance to mechanism resolution

**Purpose:** establish the estimand and distinguish detection from explanation.

**Content:** intact state A vs perturbed state B; candidate usefulness `E_A(h)` and `E_B(h)`; conditional change `Delta_h`; FALI explanation through T/R/O/V/C.

**Data source:** none; deterministic schematic derived directly from the formal specification.

**Main claim:** FALI explains a selected cross-state change in output-grounded causal usefulness; it does not perform circuit discovery.

---

## Figure 2 — FALI intervention game

**Purpose:** make the score partition and 32-cell intervention mechanics transparent.

**Content:** triangular attention-score matrix with T/R/O partition; value tensor V; context C; one example hybrid cell; masked softmax -> pattern -> z -> full candidate injection; same-context zero baseline; standard Owen structure SCORE={T,R,O}, VALUE={V}, CONTEXT={C}; 36 admissible permutations; coarse/fine reconciliation.

**Data source:** none; deterministic schematic from the formal specification and implementation semantics.

**Threat to avoid:** implying hybrids are naturally occurring states or unique causal pathways.

---

## Figure 3 — Ground-truth and implementation validation

**Purpose:** demonstrate that the method identifies known factors before empirical interpretation.

**Content:** five single-channel synthetic cases; pure T×C interaction; concise implementation-invariant dashboard.

**Primary source:** `results/extended_fali/ef1_synthetic/validation_report.json`.

**Main-text claim allowed:** the mathematical/intervention pipeline passed controlled factor-identification and implementation-invariant tests.

**Do not claim:** synthetic validation proves real-model mechanistic truth.

---

## Figure 4 — GPT-2: attention adaptation and causal recruitment are different

**Purpose:** establish the corrected functional problem and canonical GPT-2 mechanism.

**Content:** L10H2 vs L10H10 attention-change/conditional-gain contrast; T/R FALI intervals; SCORE/V/C explanation of why positive score adaptation need not imply positive total gain.

**Primary sources:**
- `results/extended_fali/ef4_canonical/aggregate/exact_means.csv`
- `results/extended_fali/ef4_canonical/aggregate/hierarchical_bootstrap_ci.csv`
- `results/extended_fali/ef4_canonical/aggregate/fine_score_gate.json`

**Locked anchors:** L10H2 conditional gain ≈0.298, phi_T≈0.783, phi_R≈−0.341; L10H10 conditional gain≈−0.263, delta target attention≈+0.310, phi_T≈+0.329, phi_R≈−0.126.

---

## Figure 5 — Principal empirical result: same recruitment label, opposite fine mechanism

**Purpose:** visual centerpiece of the paper.

**Content:** positive conditional recruitment in GPT-2 IOI and Pythia induction; GPT-2 L10H2 T+/R− versus Pythia L8H2 T−/R+; explicit EF-6 transfer-gate failure.

**GPT-2 sources:** EF-4 exact means / hierarchical CIs.

**Pythia sources:**
- `results/extended_fali/ef6_pythia_induction/aggregate/exact_means.csv`
- `bootstrap_ci.csv`
- `hypothesis_results.json`
- `ef6_gate.json`

**Visual claim:** both studied settings exhibit positive conditional recruitment, while the target/rival signs are opposite.

**Caption wording:** "Conditional recruitment occurs in both studied settings, but the fine target/rival score contributions differ in sign. The Pythia experiment therefore rejects the preregistered GPT-2 sign-transfer hypothesis."

**Do not write:** "FALI proves there is no universal repair mechanism."

---

## Supplementary Figure S1 — Intervention-scheme robustness

**Purpose:** show the tested matched alternative ablations without overstating EF-5.

**Sources:** EF-5 ablation gate, channel robustness, exact means, hierarchical CIs.

**Required annotation:** matched_mean = ROBUST; matched_resample = ROBUST; overall EF5_ABLATION_GATE = FAIL because the zero-damage reference criterion failed.

No figure may summarize this as simply "robust across ablations."

Additional supplementary figures should cover GPT-2 template/stratum heterogeneity, same-layer controls, Pythia seed/offset heterogeneity, detailed EF-1 invariants, and coarse/fine quotient reconciliation as specified in `FIGURE_PRODUCTION_SPEC.md`.

---

## Visual standards

- Vector PDF/SVG preferred for diagrams and plots.
- Plot scripts must read source CSV/JSON; no manually entered empirical heights/points/CIs.
- Show uncertainty intervals for inferential empirical quantities.
- Use shapes/line styles in addition to color so plots remain interpretable in grayscale and for color-vision deficiencies.
- Avoid legends when direct labels are clearer.
- No chart title embedded inside the figure; title-level information belongs in the caption.
- Consistent 8–10 pt final-size lettering.
- Figure filenames for submission: `Fig1.pdf` through `Fig5.pdf`.
- Full panel-by-panel rendering requirements are frozen in `FIGURE_PRODUCTION_SPEC.md`.
