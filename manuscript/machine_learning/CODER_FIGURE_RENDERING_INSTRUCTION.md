# CODER INSTRUCTION — Render FALI Main Figures 1–5

Branch: `manuscript/reconstruction`

Scientific design is frozen in:

- `manuscript/machine_learning/FIGURE_PRODUCTION_SPEC.md`
- `manuscript/machine_learning/FIGURE_PLAN.md`
- `manuscript/machine_learning/EVIDENCE_LEDGER.md`

Do not make scientific design decisions. If these documents conflict, STOP and report the conflict rather than choosing an interpretation.

## Objective

Render the five main manuscript figures as publication-quality, fully reproducible vector graphics using only committed formal specifications and committed experiment/validation outputs.

This is a rendering/implementation task. Do not rerun model experiments, alter statistics, reinterpret failed gates, select different heads, or change manuscript claims.

## 1. Global visual system

Use one coherent visual language across Figures 1–5.

Preferred channel palette (Okabe–Ito / color-vision-safe):

- T / target: blue `#0072B2`
- R / rival: vermillion `#D55E00`
- O / other scores: neutral gray `#7F7F7F`
- V / values: green `#009E73`
- C / context: purple `#CC79A7`
- endpoint / conditional gain: black

Color must never be the only encoding. Also distinguish channels with marker shape, line style, hatch, or direct text labels.

General style:

- white background;
- minimal gridlines;
- strong zero reference lines on signed plots;
- no chart titles inside figures;
- panel labels `(a)`, `(b)`, etc.;
- direct labels preferred over legends;
- final-size font approximately 8–10 pt;
- panel labels approximately 10–11 pt bold;
- no 3D effects, gradients, shadows, decorative neural-network art, or generative imagery;
- avoid default Matplotlib appearance: explicitly control spacing, typography, axis weights, marker sizes, and annotation placement;
- figures must remain interpretable in grayscale;
- use consistent channel ordering T, R, O, V, C whenever applicable.

Target width:

- double-column: approximately 174 mm for all five main figures;
- ensure labels remain legible after reduction.

Exports for every main figure:

- `FigN.pdf` — vector, manuscript source of truth;
- `FigN.svg` — editable vector;
- `FigN.png` — 300 dpi review preview.

Store under:

`manuscript/machine_learning/figures/`

Plotting/source scripts under:

`manuscript/machine_learning/figure_scripts/`

Do not manually edit generated PDF/SVG after rendering.

## 2. Data integrity rule

Figures 1–2 are deterministic method schematics and must follow the formal specification exactly.

Figures 3–5 contain validation/experimental quantities. Every point estimate, CI, sign, sample count, or gate label must be parsed from committed CSV/JSON output whenever available.

Do not type numerical plot values into scripts merely because they are listed in the production specification. The listed values are audit anchors, not a replacement for source parsing.

For each figure create a machine-readable provenance sidecar:

`FigN_provenance.json`

Record:

- figure number;
- source paths;
- source SHA256 values;
- metrics/rows extracted;
- plotting script path;
- plotting script git SHA if available;
- generated file SHA256 values;
- timestamp;
- any non-data visual constants.

If a required source path is missing or a parsed value disagrees with the frozen specification beyond normal formatting precision: STOP and report.

## 3. Figure 1 — From conditional importance to mechanism resolution

Type: deterministic vector schematic. No empirical data.

Implement exactly from `FIGURE_PRODUCTION_SPEC.md`.

Three-panel horizontal flow:

(a) State A intact vs State B perturbed, each showing candidate-present minus candidate-zero effect.

(b) `Delta_h = E_B(h) - E_A(h)` with distinction:

- detection/localization: did importance change?
- FALI: which changes account for Delta_h?

(c) T/R/O/V/C tokens feeding the exact accounting identity:

`phi_T + phi_R + phi_O + phi_V + phi_C = Delta_h`

Guardrails:

- do not draw T/R/O/V/C as five natural anatomical pathways;
- do not imply FALI discovers the candidate;
- C must read `surrounding context` or `surrounding-network context`, never `downstream sensitivity`;
- explicitly mark intervention/game semantics as counterfactual.

Preferred implementation: SVG/TikZ-equivalent deterministic vector source. Python/vector libraries are acceptable if fully reproducible.

## 4. Figure 2 — How the FALI intervention game works

Type: deterministic vector schematic. No empirical data.

Three panels:

(a) triangular causal score matrix showing T target coordinate, R remaining valid scores in final task row, O earlier valid rows, and structural mask as non-player coordinates; aligned V tensor beside it.

(b) one illustrative hybrid cell, for example `(T_B, R_A, O_B, V_A, C_B)`, flowing through:

`hybrid scores -> masked softmax -> hybrid pattern -> P @ V -> hybrid z -> full candidate-z injection`

Show candidate-zero baseline in the same C context and:

`U(cell) = metric(hybrid injected) - metric(candidate zero, same C)`

(c) `2^5 = 32 cells` -> standard Owen attribution with:

`SCORE={T,R,O}`
`VALUE={V}`
`CONTEXT={C}`

and identities:

`Phi_SCORE = phi_T + phi_R + phi_O`

`Phi_SCORE + phi_V + phi_C = Delta_h`

Add `36 admissible contiguous-block permutations` in small explanatory text.

Do not show Q/K as additional Owen children.

## 5. Figure 3 — Controlled factor recovery and implementation validation

Primary source:

`results/extended_fali/ef1_synthetic/validation_report.json`

Panel (a): five-by-five normalized attribution matrix or five horizontal attribution rows.

Rows:

- T-only
- R-only
- O-only
- V-only
- C-only

Columns:

- phi_T
- phi_R
- phi_O
- phi_V
- phi_C

Normalize each row by its own absolute `DeltaU` for visual comparability while preserving sign. Annotate raw DeltaU at row end. Parse all values from JSON.

Panel (b): pure T x C interaction. Show total DeltaU and equal Owen split between T and C; R/O/V zero. Parse from `level_a.tc_interaction`.

Panel (c): implementation-invariant dashboard. Parse recorded errors from `level_b` plus Level-A quotient bridge. Include at least:

- score -> pattern;
- pattern/value -> z;
- independent endpoint identity;
- all-A/all-B injected-z identity;
- non-candidate modification;
- live Q leak;
- live V leak;
- live Q+V leak;
- model quotient bridge;
- synthetic quotient bridge;
- overall pass.

This panel should communicate precision without visually overemphasizing repeated zeros. A compact table/check strip with scientific-notation error values is preferred over ten bars at zero.

## 6. Figure 4 — GPT-2: attention adaptation is not functional recruitment

Primary sources:

- `results/extended_fali/ef4_canonical/aggregate/hierarchical_bootstrap_ci.csv`
- `results/extended_fali/ef4_canonical/aggregate/exact_means.csv`
- `results/extended_fali/ef4_canonical/aggregate/fine_score_gate.json`

Panel (a): 2D error-bar plot.

- x = `delta_target_attn`
- y = `conditional_gain`
- heads = L10H2, L10H10 only
- x=0 and y=0 reference lines
- direct labels
- callouts:
  - L10H2: `attention up, usefulness up`
  - L10H10: `attention up, usefulness down`

Both x/y intervals must come from the canonical hierarchical source.

Panel (b): horizontal forest plot for fine channels:

1. L10H2 T
2. L10H2 R
3. L10H10 T
4. L10H10 R

Show point estimate + 95% CI and x=0 reference. T/R use both distinct color and marker shape. Read gate status from JSON and annotate unobtrusively: `EF-4 fine-score gate: PASS`.

Panel (c): coarse SCORE/V/C decomposition for L10H2 and L10H10 plus endpoint conditional gain as black diamond. Do not stack confidence intervals. Purpose: show that positive SCORE contribution can coexist with negative endpoint gain when C is sufficiently negative.

Never label L10H10 as recruited in this figure.

## 7. Figure 5 — GPT-2 versus Pythia: same recruitment label, opposite fine-score signs

This is the manuscript centerpiece. Give it the cleanest composition and largest visual emphasis.

GPT-2 sources:

- `results/extended_fali/ef4_canonical/aggregate/hierarchical_bootstrap_ci.csv`
- `results/extended_fali/ef4_canonical/aggregate/exact_means.csv`

Use L10H2 only because it satisfies the corrected positive-recruitment definition.

Pythia sources:

- `results/extended_fali/ef6_pythia_induction/aggregate/exact_means.csv`
- `results/extended_fali/ef6_pythia_induction/aggregate/bootstrap_ci.csv`
- `results/extended_fali/ef6_pythia_induction/aggregate/ef6_gate.json`
- `results/extended_fali/ef6_pythia_induction/aggregate/hypothesis_results.json`

Recommended layout:

(a) top functional strip: GPT-2 conditional gain and Pythia conditional gain, both positive with 95% CI. Do not compare magnitudes as if units are shared across tasks; separate axes or clearly state task-specific units.

(b) lower aligned signed forest plot for T/R within each setting. Use a common visual zero reference and channel colors/shapes, but do not imply absolute magnitudes are directly comparable across different task metrics. The scientific comparison is sign/organization.

GPT-2:

- T positive
- R negative

Pythia:

- T negative
- R positive

(c) optional minimal transfer-status annotation if it improves clarity:

`Recruitment: confirmed in both studied settings`

`Preregistered fine-score transfer: FAIL`

Read the gate status from source; do not manually set it.

Do not write or visually imply `no universal repair mechanism`. The supported statement is that the tested shared functional label does not imply a shared target/rival organization across these two studied settings.

## 8. Supplementary figures

Do not render supplementary figures yet unless needed for code reuse. The next approved supplementary set will include:

- EF-5 intervention-scheme robustness with explicit overall-gate failure;
- canonical template/stratum heterogeneity;
- same-layer architectural controls;
- Pythia seed/offset heterogeneity;
- detailed validation diagnostics;
- quotient reconciliation.

Main figures first.

## 9. Reproducibility / code requirements

Preferred implementation:

- Python 3;
- pandas/numpy for source parsing;
- matplotlib for quantitative figures;
- deterministic vector drawing library/TikZ/SVG for schematics;
- no seaborn dependency required;
- no stochastic layout unless a fixed seed is recorded.

Create a shared style module, e.g.:

`manuscript/machine_learning/figure_scripts/style.py`

and separate scripts:

- `render_fig1.py`
- `render_fig2.py`
- `render_fig3.py`
- `render_fig4.py`
- `render_fig5.py`

Add:

`render_all.py`

which regenerates all five figures from repo state.

All scripts must resolve paths relative to repository root and fail loudly on missing required files/metrics.

## 10. Automated validation

Add tests or a validation script that checks at minimum:

- expected source files exist;
- required metric rows are unique;
- parsed point estimates/CIs match frozen audit anchors to <= 1e-8 where the anchors exist;
- CI low <= point <= CI high;
- expected signs for locked empirical claims;
- EF-4 gate is `PASS`;
- EF-6 gate is `FAIL`;
- generated PDF/SVG/PNG files all exist and are non-empty;
- figure provenance JSON exists for all five figures;
- no empirical plotting script contains manually entered locked result values where the value can be source-parsed.

Do not change thresholds or source files to make validation pass.

## 11. Review previews

Create:

`manuscript/machine_learning/figures/review/`

Include one PNG preview per figure plus a contact sheet `main_figures_contact_sheet.png` showing Figures 1–5 at realistic manuscript scale.

The contact sheet is for research-lead visual review only, not manuscript inclusion.

## 12. Manuscript integration

DO NOT integrate figures into `main.tex` yet.

First render the complete Figure 1–5 review set and STOP for research-lead audit.

Do not alter scientific prose, captions, figure numbering, or manuscript claims during this phase.

## 13. Commit structure

Prefer:

Commit 1:
`feat: add reproducible main figure rendering`

Commit 2:
`figures: render FALI main figure review set`

Push to:

`manuscript/reconstruction`

No force-push.

## 14. Final response

Return:

1. final HEAD SHA;
2. files added/modified;
3. plotting/rendering dependencies used;
4. validation checks passed/failed;
5. exact source paths used for Figures 3–5;
6. generated PDF/SVG/PNG paths for Figures 1–5;
7. provenance JSON paths;
8. contact-sheet path;
9. any source/spec mismatch;
10. any typography/layout issue at 174 mm width;
11. confirmation that no model experiments were rerun;
12. confirmation that no manuscript prose was changed.

STOP.

Do not integrate figures into the manuscript until research-lead review is complete.