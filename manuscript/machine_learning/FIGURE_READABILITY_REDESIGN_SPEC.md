# FALI Main Figures — Readability Redesign Specification

Status: **SCIENTIFIC/VISUAL DESIGN FROZEN FOR REDESIGN PASS**

Purpose: replace the current visually compressed main figures with publication-readable versions at Springer two-column width (~174 mm). Scientific content, empirical values, confidence intervals, gates, and captions remain unchanged unless this specification explicitly changes visual presentation.

## 1. Global requirement

The existing main figures are scientifically valid but visually unacceptable at manuscript scale. The redesign must optimize for **actual PDF readability**, not contact-sheet readability or nominal vector resolution.

### Mandatory typography

At final 174-mm width:

- minimum text size anywhere in a figure: **8.0 pt**;
- preferred ordinary labels: **8.5–9 pt**;
- axis labels: **8.5–9 pt**;
- tick labels: **8 pt**;
- panel letters `(a)`, `(b)`, etc.: **11 pt bold**;
- small qualifying notes: **8 pt minimum**;
- no 6–7.5 pt text anywhere.

Do not solve collisions by reducing font size.

### Mandatory spacing

- no text may overlap another text element, marker, error bar, axis label, or panel boundary;
- no label should be placed immediately after a CI endpoint unless at least ~12 pt of rendered whitespace is guaranteed;
- prefer y-axis row labels, legends, or dedicated text columns over free-floating annotations;
- allow figures to become substantially taller;
- page count may increase. Readability is more important than the previous 21-page count.

### Canvas guidance

All figures remain 174 mm wide. Approximate target heights:

- Fig. 1: 80–90 mm
- Fig. 2: 95–110 mm
- Fig. 3: 100–115 mm
- Fig. 4: 100–115 mm
- Fig. 5: 120–140 mm

These are guides, not hard maxima. Never compress below readability.

### Export QA

Review must be performed on:

1. standalone figure PNG/PDF;
2. a 174-mm print-size contact sheet;
3. the **compiled manuscript PDF**, because successful standalone rendering is insufficient.

A figure is not accepted if labels overlap in `main.pdf` even if the standalone PNG looks acceptable.

---

# Figure 1 — Conditional importance to mechanism resolution

## New layout

Use **two rows** rather than three cramped equal-width columns.

Top row:
- panel (a): ~58% width;
- panel (b): ~42% width.

Bottom row:
- panel (c): full width.

## Panel (a)

Show State A and State B as two stacked rows.

Do **not** print the full long equations
`E_A(h) = m(A, h present) - m(A, h zero)` and
`E_B(h) = ...`
inside a narrow side area.

Instead show concise endpoint labels:

- `Candidate usefulness: E_A(h)`
- `Candidate usefulness: E_B(h)`

The caption/text defines the subtraction.

Keep candidate head `h` visually consistent across states and mark only the primary component as intervened in B.

## Panel (b)

Large central equation:

`Delta_h = E_B(h) - E_A(h)`

Below it use two short stacked statements:

- `Detection: did usefulness change?`
- `FALI: what accounts for the change?`

No long horizontal sentence.

## Panel (c)

Five factor boxes should be arranged in one spacious row or a 3+2 arrangement if needed.

Each factor gets:

- large symbol T/R/O/V/C;
- one short label underneath.

Below factors:

`phi_T + phi_R + phi_O + phi_V + phi_C = Delta_h`

Then one 8-pt note:

`Controlled counterfactual accounting; not five unique natural pathways.`

---

# Figure 2 — FALI intervention game

## New layout

Use a **2 x 2 conceptual grid**:

- panel (a): upper-left, approximately 48% width;
- panel (b): upper-right, approximately 52% width;
- panel (c): full-width bottom row.

Panel (c) should no longer be squeezed into one-third width.

## Panel (a): score partition

Keep the complete 6 x 6 matrix and the corrected partition:

- final row: `R R T R R R`;
- all earlier valid cells: O;
- upper triangle: structural mask.

Requirements:

- cell symbols at >=8 pt;
- `q*` and `k*` labels at >=8 pt;
- axis labels `query q` and `key k` at >=8 pt;
- structural-mask annotation placed outside the matrix with a short leader line;
- value tensor V large enough to identify without crowding.

Do not place explanatory text inside cells beyond T/R/O.

## Panel (b): hybrid construction

Use a vertical flow with **short boxes only**:

`Hybrid scores`
→ `Masked softmax`
→ `Pattern P`
→ `P @ V`
→ `Candidate z`
→ `Inject full z`

Place the illustrative state choice at top in compact math-like form:

`T_B  R_A  O_B  V_A  C_B`

At bottom use a two-line utility definition, not one long line:

`U(cell) = metric(hybrid candidate)`
`          - metric(candidate zero, same C)`

Use >=8 pt throughout.

## Panel (c): game + Owen bridge

Use the full width and create three visual blocks:

Left:
`32 hybrid cells`

Center:
- `SCORE = {T,R,O}`
- `VALUE = {V}`
- `CONTEXT = {C}`

Right:
- `Phi_SCORE = phi_T + phi_R + phi_O`
- `Phi_SCORE + phi_V + phi_C = Delta_h`

Bottom note:

`Standard Owen value; 36 admissible block-contiguous permutations.`

Second short note:

`Mixed cells are controlled counterfactuals.`

---

# Figure 3 — Controlled validation

## New layout

Use **two rows**:

Top row:
- panel (a): ~58% width;
- panel (b): ~42% width.

Bottom row:
- panel (c): full width.

## Panel (a): single-channel recovery

Keep the 5 x 5 signed matrix.

Requirements:

- row labels >=8.5 pt;
- column labels use actual mathtext or clean Unicode `phi_T`, etc., >=8.5 pt;
- cell numbers >=8 pt;
- raw Delta U values in a dedicated right-side column >=8 pt;
- colorbar optional; if retained, must not reduce matrix readability;
- if colorbar causes crowding, remove it and rely on fixed `[-1,1]` normalization plus cell values.

The matrix is the main object of panel (a), not the colorbar.

## Panel (b): T x C interaction

Use a simple horizontal contribution plot with rows:

- total Delta U
- phi_T
- phi_C
- other channels (combined or visibly zero)

Do not devote separate visual space to three zero bars if doing so reduces readability.

Prominent note:

`T x C interaction -> 50/50 Owen split`

>=8 pt.

## Panel (c): implementation invariants

Replace the long tiny dashboard with a **grouped validation table/checklist**.

Use 6 grouped rows maximum:

1. `Score -> pattern reconstruction` — max error
2. `Pattern/value -> z reconstruction` — max error
3. `Mask + batch partition` — PASS / max error
4. `Independent endpoints` — max error
5. `Candidate isolation / live leakage` — max error
6. `Coarse-fine quotient bridge` — max error

At bottom:

`Overall validation: PASS`

All text >=8 pt.

The exact detailed checks remain in source/provenance/supplement and do not need individual rows in the main figure.

---

# Figure 4 — GPT-2 attention adaptation vs recruitment

## New layout

Use **two rows**:

Top row:
- panel (a): full width, but plot itself may occupy ~60–70% centered width.

Bottom row:
- panel (b): ~50% width;
- panel (c): ~50% width.

This figure must not use free-floating explanatory phrases beside points or CIs.

## Panel (a)

Keep x = change in target attention, y = conditional gain.

Plot only:

- L10H2
- L10H10

Use distinct marker shapes.

Direct labels should be only:

- `L10H2`
- `L10H10`

Remove in-axis phrases:

- `attention up, usefulness up`
- `attention up, usefulness down`

The quadrant and caption communicate that message.

Use strong x=0 and y=0 references.

## Panel (b): fine T/R forest

Use y-axis labels rather than text after CI endpoints:

- `L10H2  T`
- `L10H2  R`
- `L10H10 T`
- `L10H10 R`

No labels after error bars.

Use T/R marker/color distinctions.

Place `EF-4 fine-score gate: PASS` as a dedicated note below or above the plotting region, not inside the data area.

## Panel (c): SCORE / V / C / gain

Use four y rows:

- SCORE
- V
- C
- endpoint gain

Plot both heads using different marker shapes with a compact legend:

- circle = L10H2
- triangle = L10H10

Do not write `L10H2 context`, `L10H10 context`, or endpoint labels next to individual data points.

The legend carries head identity.

Zero line prominent.

---

# Figure 5 — Cross-setting mechanism transfer

## New layout

Use **three spacious rows**.

### Row 1: panel (a)

Two task-specific conditional-gain mini-panels side by side.

Each mini-panel should be wide enough for its y-axis title and CI.

Titles:

- `GPT-2 IOI — L10H2`
- `Pythia induction — L8H2`

Do not add a global text label that competes with panel labels.

A short note centered below the pair may say:

`Positive conditional gain in both studied settings`

>=8.5 pt.

### Row 2: panel (b) — visual centerpiece

Two side-by-side forest panels with significantly more vertical and horizontal space than current version.

Each has exactly two rows:

- phi_T
- phi_R

Titles:

- `GPT-2 IOI`
- `Pythia induction`

Use zero lines and large markers/error bars.

Remove extra `T +`, `R -`, `T -`, `R +` text inside the axes. The plotted signs make this visible.

Centered below:

`Opposite target/rival signs`

Do not compare raw attribution magnitudes across tasks.

### Row 3: panel (c)

Keep the five-row gate table, but make it taller.

Columns:

- Gate
- Preregistered criterion
- Outcome

Rows G0–G4.

Font >=8 pt, preferably 8.5 pt.

Below table:

`EF-6 transfer gate: FAIL`

Then:

`Mechanism-transfer failure; recruitment remains confirmed.`

Do not use warning/error styling.

---

# 6. Compiled manuscript integration QA

After rerendering, compile `main.tex` without changing figure widths below `\textwidth` unless required by clipping.

For every figure page, inspect at 100% PDF zoom and at approximately normal print scale.

A figure fails QA if any of the following occurs:

- any text overlap;
- any label touches an error bar/marker;
- any panel title overlaps panel letter;
- any text smaller than 8 pt at final size;
- T/R/O matrix labels indistinguishable;
- figure requires zooming substantially beyond normal reading size to understand labels;
- table text is unreadable;
- figure is vertically compressed merely to reduce page count.

Page count may increase. Do not optimize page count until figures are readable.

# 7. Scientific invariants

No empirical values, CIs, gates, signs, head identities, target definitions, or captions may change in this redesign pass.

Required statuses remain:

- EF-1R PASS;
- EF-4 PASS;
- EF-5 overall FAIL;
- EF-6 FAIL;
- GPT-2 L10H2: T positive, R negative;
- GPT-2 L10H10: T positive, R negative, total gain negative;
- Pythia L8H2: T negative, R positive, recruitment positive.

This is a visual redesign only.