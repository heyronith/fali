# Figure Production Specification — FALI / Machine Learning

Status: **SCIENTIFIC DESIGN FROZEN FOR RENDERING**  
Target: **5 main figures** + supplementary robustness/diagnostic figures.  
Scientific owner: research lead. Rendering owner: coding agent.

## Global design rules

- Figures 1–2 are deterministic **method schematics**, not empirical plots. They must implement the formal FALI definition exactly.
- Figures 3–5 contain **real validation/experimental quantities** read directly from committed CSV/JSON outputs.
- No bar height, point estimate, CI, sign, or sample count may be manually typed into plotting code when it can be read from source data.
- Plot scripts must emit vector PDF/SVG and a PNG preview.
- Use one visual language throughout: `T` target, `R` rival, `O` other score, `V` value, `C` context. Distinguish channels by both color and shape/pattern so grayscale remains readable.
- Zero reference lines must be visually prominent in all signed-attribution plots.
- Inferential quantities show the exact preregistered 95% intervals used in the manuscript.
- Direct labels are preferred to legends when practical.
- No embedded chart titles; panel labels `(a)`, `(b)`, etc. only. Figure-level explanation belongs in the caption.
- Final-size text: approximately 8–10 pt; panel labels 10–11 pt bold.
- Springer target widths: single column ~84 mm; double column ~174 mm. Figures 1, 2, 4, 5 are designed for double-column width; Figure 3 may be double-column for legibility.
- No decorative/generative imagery and no neural-network clip art.

---

# Figure 1 — From conditional importance to mechanism resolution

**Role in paper:** establish the scientific question before mathematical detail.  
**Type:** deterministic vector schematic.  
**Experimental data:** none.

## Visual objective

A reader should understand in <15 seconds:

1. a primary circuit component is perturbed;
2. a candidate head's causal usefulness changes;
3. ordinary conditional attribution tells us **that** it changed;
4. FALI asks **why** it changed.

## Layout

Horizontal 3-panel flow, approximately 174 mm × 65–75 mm.

### (a) Two circuit states

Show two compact rows:

- **State A — intact**: primary component present; candidate head `h` present.
- **State B — perturbed**: primary component marked as intervened/removed; same candidate head highlighted.

At right of each row show the candidate-effect definition:

`E_A(h) = metric(A, h present) − metric(A, h zero)`

`E_B(h) = metric(B, h present) − metric(B, h zero)`

Do not depict `B` as globally “broken”; label it only as the **perturbed context**.

### (b) Conditional change

Center a large signed difference:

`Δ_h = E_B(h) − E_A(h)`

Below it:

- `Detection/localization: did importance change?`
- `FALI: which changes account for Δ_h?`

Use an arrow from the scalar `Δ_h` into panel (c).

### (c) Mechanism-resolution factors

Show five factor tokens feeding an accounting bracket:

- `T` — target score(s)
- `R` — rival scores
- `O` — other valid scores
- `V` — value content
- `C` — surrounding context

Then the exact accounting identity:

`φ_T + φ_R + φ_O + φ_V + φ_C = Δ_h`

Small footer:

`Attributions are defined under an explicit counterfactual intervention game.`

## Visual guardrails

- Do **not** draw T/R/O/V/C as five natural anatomical pathways.
- Do **not** imply that FALI discovers the candidate head.
- Do **not** call C “downstream sensitivity.”
- Candidate head and task-relevant edge are inputs to the method.

## Source of truth

- `manuscript/machine_learning/main.tex`, Sec. Problem Formulation
- `docs/EXTENDED_FALI_FORMAL_SPEC.md`
- `manuscript/machine_learning/EVIDENCE_LEDGER.md`, M1–M7

## Draft caption

**Figure 1 | FALI explains a conditional change rather than detecting it.** A selected attention head has output-grounded causal usefulness `E_A(h)` in an intact context and `E_B(h)` after a prespecified circuit perturbation. Their difference `Δ_h` is the conditional gain to be explained. FALI constructs controlled hybrid interventions over target scores (`T`), rival scores (`R`), other valid scores (`O`), value content (`V`), and surrounding-network context (`C`), and allocates the endpoint change among these factors. The method is hypothesis-driven: the candidate head and task-relevant target are specified before FALI analysis.

---

# Figure 2 — How the FALI intervention game works

**Role in paper:** make the 32-cell construction and coarse/fine attribution intuitive.  
**Type:** deterministic vector schematic.  
**Experimental data:** none.

## Visual objective

A reader should be able to trace one hybrid cell from cached endpoint states to output-grounded utility and understand why `T/R/O` form the SCORE coalition.

## Layout

Three panels, left-to-right.

### (a) Attention-score partition

Draw a triangular causal attention-score matrix for one candidate head.

Mark:

- final task query row `q*`;
- target coordinate `(q*, k*)` as `T`;
- remaining structurally valid coordinates in final row as `R`;
- all earlier structurally valid query rows as `O`;
- future/padding coordinates cross-hatched as **structural mask — not players**.

Beside matrix show full value tensor `V` as a vertical stack aligned with key/source positions.

### (b) One hybrid cell

Show two cached endpoints at top:

`A: S^A, V^A` and `B: S^B, V^B`.

Below, illustrate an example cell, e.g. `(T_B, R_A, O_B, V_A, C_B)`, purely as an example of mechanics—not a favored scientific condition.

Pipeline:

`hybrid scores` → `masked softmax` → `hybrid pattern` → `P @ V` → `hybrid z` → `inject full candidate z`.

Parallel zero-baseline branch:

`same C context` → `candidate z = 0`.

Then:

`U(cell) = metric(hybrid injected) − metric(candidate zero, same C)`.

### (c) 32 cells → standard Owen attribution

Show a compact 5-bit cube/table icon labelled `2^5 = 32 cells`.

Then coalition structure:

`SCORE = {T,R,O}`

`VALUE = {V}`

`CONTEXT = {C}`

Show:

`Φ_SCORE = φ_T + φ_R + φ_O`

and

`Φ_SCORE + φ_V + φ_C = Δ_h`.

Add small text:

`standard Owen value; 36 admissible contiguous-block permutations`.

## Visual guardrails

- Mixed cells are counterfactual and may be off-manifold.
- Never call the 32 cells “natural states.”
- Never show Q and K as additional Owen children; Q/K remain secondary diagnostics, not this hierarchy.

## Source of truth

- `manuscript/machine_learning/main.tex`, Sec. FALI
- `docs/EXTENDED_FALI_FORMAL_SPEC.md`
- `src/extended_fali_interventions.py`
- `src/extended_fali_math.py`

## Draft caption

**Figure 2 | FALI constructs an output-grounded 32-cell intervention game.** Structurally valid attention scores are partitioned into the task-target coordinate(s) `T`, rivals `R` in the same query row, and all other valid scores `O`; `V` denotes the candidate value tensor and `C` the surrounding intact or perturbed computation. Each hybrid score tensor is passed through the model's masked softmax, combined with the selected values, and injected as the candidate's full head output. Utility is the task metric relative to a candidate-zero run in the same context. Standard Owen attribution under `SCORE={T,R,O}`, `VALUE={V}`, `CONTEXT={C}` allocates interactions while reconciling the fine score attributions with the coarse three-factor game.

---

# Figure 3 — Controlled factor recovery and implementation validation

**Role in paper:** establish that FALI's implementation recovers known manipulated factors before pretrained-model interpretation.  
**Type:** empirical/validation figure; all values from EF-1R.  
**Primary source:** `results/extended_fali/ef1_synthetic/validation_report.json`.

## Layout

Three panels.

### (a) Five single-channel ground-truth cases

Use a **5 × 5 attribution matrix/heatmap** or five small horizontal stacked rows.

Rows = synthetic case manipulated: `T only`, `R only`, `O only`, `V only`, `C only`.

Columns = recovered attribution: `φ_T`, `φ_R`, `φ_O`, `φ_V`, `φ_C`.

Values must be read from `level_a.single_channel`.

Recommended visual encoding:

- each row normalized by its own `|ΔU|` for visual comparability;
- annotate the raw `ΔU` at row end;
- diagonal should be ±1 after normalization, all off-diagonals zero at recorded precision;
- retain sign, especially the R-only case which has negative `ΔU`.

Raw endpoint changes:

- T-only: `+0.2689520684`
- R-only: `−0.4940997625`
- O-only: `+0.2134890065`
- V-only: `+0.1925102705`
- C-only: `+0.7112345942`

Do not hard-code these in plotting logic; parse JSON.

### (b) Known T×C interaction

Two interval-free points/bars because this is deterministic ground truth:

- total `ΔU ≈ 2.0`;
- recovered `φ_T ≈ 1.0`;
- recovered `φ_C ≈ 1.0`;
- R/O/V = 0.

Visually show `50% / 50%` split.

Source: `level_a.tc_interaction`.

This panel is important because it demonstrates that standard Owen allocation does not arbitrarily dump the cross-block interaction into context.

### (c) Implementation-invariant dashboard

Compact table/check-strip parsed from `level_b` plus the Level-A quotient value. Suggested rows:

| Check | Recorded max error |
|---|---:|
| score → pattern, A/B | `0` |
| pattern/value → z, A/B | `0` |
| independent endpoint identity | `0` |
| all-A / all-B injected-z identity | `0` |
| non-candidate head modification | `0` |
| live Q leak | `0` |
| live V leak | `0` |
| live Q+V leak | `0` |
| model-level quotient bridge | `0` |
| synthetic quotient bridge | `1.11×10^-16` |

The code should read fields rather than type values manually.

Footer: `overall_pass = true`.

## Scientific claim supported

The controlled systems and implementation checks support that the coded intervention game recovers the intended manipulated channel and satisfies the preregistered identities. They do **not** establish that the factorization is uniquely true of natural transformer computation.

## Draft caption

**Figure 3 | FALI recovers controlled ground truth and satisfies implementation invariants.** (a) In five deterministic attention systems where only one of `T`, `R`, `O`, `V`, or `C` differs between endpoints, the entire endpoint change is assigned to the manipulated factor up to floating-point error. Rows are normalized by their own absolute endpoint change for visual comparison; raw `ΔU` is annotated. (b) A constructed pure `T×C` interaction yields an equal Owen split between `T` and `C`, as analytically expected. (c) A randomly initialized two-layer TransformerLens model independently validates score-to-pattern and pattern/value-to-output reconstruction, endpoint identity, batch-safe partitions, candidate-isolation, live-computation leakage checks, and coarse--fine quotient reconciliation. These are implementation/ground-truth tests, not evidence that FALI's hybrid factors are unique natural pathways.

---

# Figure 4 — GPT-2: attention adaptation is not functional recruitment

**Role in paper:** establish the corrected functional problem and show what FALI adds in the canonical GPT-2 setting.  
**Type:** empirical; real EF-4 results.  
**Primary source:** `results/extended_fali/ef4_canonical/aggregate/hierarchical_bootstrap_ci.csv`.  
**Supporting source:** `results/extended_fali/ef4_canonical/aggregate/exact_means.csv` and `fine_score_gate.json`.

## Layout

Three panels.

### (a) Attention change versus conditional gain

Use a 2D point plot with error bars:

- x-axis: `Δ target attention`
- y-axis: `conditional gain`
- horizontal zero line for gain;
- vertical zero line for attention change.

Plot only:

- L10H2
- L10H10

Both x and y error bars use their hierarchical 95% intervals from the same EF-4 source.

Exact source values:

**L10H2**
- Δ target attention: `0.27946`, CI `[0.26164, 0.29551]`
- conditional gain: `0.29845`, CI `[0.26294, 0.33518]`

**L10H10**
- Δ target attention: `0.31025`, CI `[0.28528, 0.33454]`
- conditional gain: `−0.26344`, CI `[−0.30191, −0.22487]`

Direct-label points. Add short callouts:

- L10H2: `attention ↑, usefulness ↑`
- L10H10: `attention ↑, usefulness ↓`

This should be the intuitive “attention ≠ recruitment” panel.

### (b) Fine-score FALI decomposition

Horizontal forest/interval plot for `φ_T` and `φ_R` for L10H2 and L10H10.

Order rows:

1. L10H2 — T
2. L10H2 — R
3. L10H10 — T
4. L10H10 — R

Show point estimate and 95% CI with a strong x=0 reference.

Values read from EF-4 CI file:

- L10H2 `φ_T = 0.78281` `[0.72186, 0.83834]`
- L10H2 `φ_R = −0.34142` `[−0.37167, −0.31091]`
- L10H10 `φ_T = 0.32852` `[0.30020, 0.35504]`
- L10H10 `φ_R = −0.12591` `[−0.13861, −0.11305]`

Use shape as well as color to distinguish T vs R.

Small note below: `EF-4 fine-score gate: PASS` (read from gate JSON).

### (c) Why favorable score adaptation need not imply recruitment

Coarse `SCORE / V / C` interval plot for the same two heads.

Rows or grouped points:

**L10H2**
- SCORE `+0.44525` `[0.40880, 0.47711]`
- V `−0.00035` `[−0.00081, 0.00010]`
- C `−0.14645` `[−0.16469, −0.12832]`
- endpoint conditional gain `+0.29845`

**L10H10**
- SCORE `+0.20625` `[0.18509, 0.22566]`
- V `+0.00156` `[0.00102, 0.00209]`
- C `−0.47125` `[−0.51427, −0.42747]`
- endpoint conditional gain `−0.26344`

Recommended encoding: three signed component points/intervals per head plus a separate diamond for the endpoint gain. Do not stack CIs.

This panel gives the mechanism-level reason L10H10 can have positive SCORE adaptation yet negative total gain.

## Statistical note

The canonical design contains 15 low-level templates × BABA/ABBA × 30 prompts × four prompt seeds = 3600 prompts. CIs are the preregistered hierarchical template-cluster intervals used in the manuscript.

## Draft caption

**Figure 4 | Target-attention adaptation and positive functional recruitment are distinct in GPT-2 IOI.** (a) L10H2 and L10H10 both increase attention to the IO target after the canonical primary Name Movers are ablated, but their conditional gains have opposite signs: L10H2 becomes more causally useful whereas L10H10 becomes less useful on the broad canonical distribution. (b) Both heads nevertheless show positive target-score and negative rival-score FALI contributions; the preregistered EF-4 fine-score gate passes. (c) Coarse attribution clarifies the distinction: L10H10's positive SCORE contribution is outweighed by a substantially negative surrounding-context contribution, producing a negative endpoint change. Points are exact balanced means; intervals are hierarchical 95% bootstrap CIs over the canonical template design.

---

# Figure 5 — Principal result: same recruitment label, opposite fine-score mechanisms

**Role in paper:** visual centerpiece / take-home empirical result.  
**Type:** empirical cross-setting comparison.  
**GPT-2 source:** `results/extended_fali/ef4_canonical/aggregate/hierarchical_bootstrap_ci.csv`.  
**Pythia source:** `results/extended_fali/ef6_pythia_induction/aggregate/exact_means.csv`, `bootstrap_ci.csv`, `hypothesis_results.json`, `ef6_gate.json`.

## Layout

Two main panels plus a narrow synthesis strip.

### (a) Functional phenomenon exists in both settings

Two point-and-interval estimates for **conditional gain**:

- GPT-2 IOI, L10H2 after canonical primary Name-Mover knockout:
  `+0.29845`, CI `[0.26294, 0.33518]`

- Pythia-160M induction, L8H2 after L5H0 knockout:
  `+0.48565`, CI `[0.32649, 0.72182]`

Because the task metrics differ in units, **do not put them on one shared numeric y-axis as if magnitudes are comparable**. Use two aligned mini-axes or a sign/status layout, each with its own numeric scale and explicit metric label:

- GPT-2: IO−S logit-difference units
- Pythia: target log-probability units

Common annotation above: `positive conditional recruitment`.

### (b) Fine T/R mechanism reverses

This panel should use one shared attribution axis if legible, because both are FALI utility contributions but task metrics still differ in scale; the visual purpose is sign comparison, not magnitude comparison.

Recommended design: two side-by-side forest plots with identical visual geometry and zero-centered axes, each numerically labeled.

**GPT-2 L10H2**
- `φ_T = +0.78281` CI `[+0.72186, +0.83834]`
- `φ_R = −0.34142` CI `[−0.37167, −0.31091]`

**Pythia L8H2**
- `φ_T = −0.14374` CI `[−0.24744, −0.08408]`
- `φ_R = +0.11780` CI `[+0.05822, +0.21913]`

Place T above R in both mini-panels. Use the same T/R shapes/colors.

Add directional arrows in the whitespace:

GPT-2: `T +` / `R −`

Pythia: `T −` / `R +`

Do not normalize away the raw attributions in the primary plot. If a small inset is desired, the preregistered Pythia absolute-score shares are `54.9%` T and `45.0%` R from denominator `0.261807`; this inset is optional and must not be compared to a differently defined GPT-2 percentage without a matched calculation.

### (c) Preregistered transfer verdict

Narrow bottom strip:

`G0 behavior ✓   G1 primary effect ✓   G2 recruitment ✓   G3 T-sign ✗   G4 R-sign ✗`

Then:

`EF-6 transfer gate: FAIL`

Small explanatory text:

`Failure is mechanistic: both fine-score signs reverse; recruitment remains confirmed.`

All statuses must be parsed from `ef6_gate.json` / `hypothesis_results.json`.

## Scientific claim supported

Allowed:

`Conditional recruitment occurs in both studied settings, while the preregistered fine-score mechanism does not transfer.`

Not allowed:

- `There is no universal repair mechanism.`
- `Recruitment generally reverses across architectures.`
- direct magnitude comparison of GPT-2 and Pythia attributions as though task metrics share units.

## Draft caption

**Figure 5 | Conditional recruitment occurs in both studied settings, but the preregistered fine-score mechanism reverses.** (a) GPT-2 IOI L10H2 and Pythia-160M induction L8H2 both have positive conditional gain after their respective primary interventions; axes are kept task-specific because the metrics use different units. (b) Their FALI target/rival decompositions have opposite signs: GPT-2 is target-positive/rival-negative, whereas Pythia is target-negative/rival-positive. (c) Accordingly, the preregistered Pythia transfer test confirms induction behavior, primary-head importance, and conditional recruitment (G0–G2) but fails both sign-transfer gates (G3–G4), so EF-6 is a failed mechanism-transfer hypothesis rather than a failed recruitment result. Intervals are the preregistered 95% bootstrap CIs for each experiment.

---

# Supplementary figures

These should be produced after the five main figures are locked.

## Supplementary Figure S1 — EF-5 intervention-scheme checks

Show zero / matched-mean / matched-resample fine channels for L10H2 and L10H10.

Mandatory annotation:

- `matched_mean: ROBUST`
- `matched_resample: ROBUST`
- `overall EF5_ABLATION_GATE: FAIL`
- reason: zero-damage reference criterion failed.

Never title or caption this as “robust across ablations.”

Sources:

- `results/extended_fali/ef5_ablation_robustness/aggregate/ablation_gate.json`
- `channel_robustness.json`
- `exact_means.csv`
- `hierarchical_bootstrap_ci.csv`

## Supplementary Figure S2 — GPT-2 stratum heterogeneity

30 canonical template-pattern stratum means for L10H2/L10H10 `φ_T` and `φ_R`, plus T0-excluded sensitivity.

Sources:

- EF-4 `stratum_means.csv`
- `strict_nonoverlap28_results.csv`
- `leave_one_template_out.csv`

## Supplementary Figure S3 — Same-layer architectural controls

L9H7 and L9H0: zero T/R/O/V attribution and endpoint change assigned to C under this intervention family.

## Supplementary Figure S4 — Pythia confirmation heterogeneity

Seed and offset breakdowns for conditional gain, `φ_T`, and `φ_R`.

Sources:

- EF-6 `seed_means.csv`
- `offset_means.csv`

## Supplementary Figure S5 — Full EF-1 validation details

Batch partition examples, structural mask checks, endpoint direct-vs-hybrid identity, Q/V/QV leak tests.

## Supplementary Figure S6 — Coarse/fine quotient bridge

Explicit numeric reconciliation of `φ_T+φ_R+φ_O` with SCORE and the historical P×V×C quotient cells.

---

# Rendering deliverables expected from coding agent

Create reproducible code under something like:

`manuscript/machine_learning/figures/scripts/`

and outputs:

- `Fig1.pdf`, `Fig1.svg`, `Fig1.png`
- `Fig2.pdf`, `Fig2.svg`, `Fig2.png`
- `Fig3.pdf`, `Fig3.svg`, `Fig3.png`
- `Fig4.pdf`, `Fig4.svg`, `Fig4.png`
- `Fig5.pdf`, `Fig5.svg`, `Fig5.png`

Each plotting script should:

1. read committed source files by repository-relative path;
2. assert expected rows/metrics/head names before plotting;
3. fail loudly if a source value is missing;
4. write a small `figure_data_manifest.json` recording source paths, source SHA256 where practical, extracted values, and output filenames;
5. contain no manually entered empirical bar heights/point estimates/CIs;
6. use deterministic layout/export settings.

Conceptual Figures 1–2 should have deterministic vector source committed alongside their outputs.

The research lead must audit the generated figures against this specification before integration into `main.tex`.
