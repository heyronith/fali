# Manuscript Plan — Machine Learning

Status: **FROZEN FOUNDATION**
Target journal: **Machine Learning (Springer Nature)**
Internal length target: **20 pages including references**
Hard ceiling without research-lead approval: **22 pages including references**

## Paper identity

**Title:** FALI: Mechanistic Decomposition of Conditional Attention-Head Importance

**One-sentence thesis:**

> FALI does not discover that an attention head became conditionally important; it explains, under an explicit intervention model, which changes in and around that head account for the change, and shows that apparently similar conditional recruitment can arise from opposite fine-grained score mechanisms.

**Empirical headline:**

> Conditional recruitment occurs in both studied settings; the fine target/rival mechanism is not shared.

Avoid the stronger universal phrasing unless explicitly qualified to the two studied settings.

## Frozen main-text architecture

### 1. Introduction — 1.75–2.0 pages

Purpose:
- establish that component importance can change after circuit intervention;
- distinguish detecting conditionality from explaining mechanism;
- introduce FALI as mechanism resolution for a preselected attention head;
- preview validation and the GPT-2/Pythia reversal;
- state exactly three contributions.

Must not begin from the old L10H10 attention-spike story.

### 2. Related Work — 1.5–1.75 pages

#### 2.1 Self-repair and conditional component importance
Hydra, Copy Suppression, broader self-repair, CoAx.
End-point: these works establish that importance/behavior can change after intervention; FALI takes that change as the explanandum.

#### 2.2 Interaction-aware causal attribution
Activation/path patching, Causal Head Gating, Curse of Multiple Mediators.
End-point: these establish context dependence/interactions; FALI seeks architecture-level resolution for a selected attention head.

#### 2.3 Decomposing attention computation
QK/OV, attention-causal communication, causal QK/value work, Owen/Shapley machinery.
End-point: routing/content decomposition and Owen mathematics are prior art; FALI's object is the cross-state change in output-grounded causal usefulness, with a surrounding-context factor and fine target/rival/other-score resolution.

### 3. Problem Formulation — 1.25–1.5 pages

#### 3.1 Conditional causal usefulness
Define states A/B, candidate h, task metric, intact effect, conditional effect, conditional gain.

#### 3.2 Mechanism-resolution objective
State assumptions:
- candidate head already identified;
- task-relevant target edge/set specified;
- goal is explanatory decomposition, not discovery.

### 4. Factorized Attention-Level Intervention — 2.75–3.25 pages

#### 4.1 Architecture-aligned factors
T, R, O, V, C.

#### 4.2 Counterfactual intervention game
Define all 32 cells and output-grounded utility compactly.
Do not enumerate every cell in main text.

#### 4.3 Standard Owen attribution and coarse/fine reconciliation
Partition SCORE={T,R,O}, VALUE={V}, CONTEXT={C}; efficiency; quotient bridge.
No claim of new Shapley/Owen mathematics.

Main text should contain only the equations needed to understand the estimand and identities. Full derivation goes to supplement.

### 5. Method Validation — 1.25–1.5 pages

#### 5.1 Analytical/synthetic ground truth
Five single-channel cases + cross-factor interaction.

#### 5.2 Transformer implementation invariants
Summarize endpoint reconstruction, batch-safe masks, score→pattern, pattern/value→z, no-live-leak, untouched-head, quotient bridge.
Full thresholds/tables to supplement.

### 6. GPT-2 IOI — 2.5–2.75 pages

#### 6.1 Experimental design
Canonical primary heads L9H6/L9H9/L10H0; canonical template design; four seeds; confirmatory head panel; IO−S metric.

#### 6.2 Conditional recruitment is not attention increase
Use the corrected functional definition. L10H2 is the main genuine-recruitment example; L10H10 is the contrasting attention-adaptation/downregulation example.

#### 6.3 Canonical fine-score mechanism
Main result: L10H2 and L10H10 show positive T and negative R in canonical EF-4; O is descriptive.

#### 6.4 Intervention robustness
Accurate wording only: matched-mean and matched-resample schemes retained the fine-channel result, while the preregistered overall EF-5 gate failed because its zero-damage reference criterion was not satisfied.

### 7. Cross-Setting Test: Pythia Induction — 1.5–1.75 pages

#### 7.1 Preregistered transfer design
Pythia-160M induction; L5H0 primary; L8H2 candidate; explain frozen transfer hypothesis.

#### 7.2 Recruitment generalizes but fine-score signs reverse
Conditional effect/gain positive; T negative; R positive; mechanism-transfer gate fails.
Treat failure as a first-class result, not a post-hoc success criterion.

### 8. Discussion — 1.0–1.25 pages

#### 8.1 Same functional label, different mechanism
Primary interpretation.

#### 8.2 Relationship to conditional attribution and patching
Compact framing:
- localization: where;
- conditional attribution: when importance changes;
- FALI: why under the specified intervention model.

### 9. Limitations — 0.75–1.0 page

Required items:
- hypothesis-driven candidate/target selection;
- hybrids may be off-manifold;
- C is coarse and not a unique localized mechanism;
- attribution depends on intervention semantics and coalition structure;
- only two substantive model/task settings;
- no universal repair-mechanism claim.

### 10. Conclusion — 0.35–0.5 page

One compact paragraph on method; one on empirical lesson.
No new claims.

### Statements/Declarations — ~0.25 page

Competing interests, funding, data/code availability, author contributions, and required AI-use disclosure based on final workflow.

### References — target 3–4 pages

Use only cited, verified references. Prefer primary papers over surveys for precedence claims.

## Main-text page budget

| Component | Target pages |
|---|---:|
| Introduction | 1.75–2.0 |
| Related Work | 1.5–1.75 |
| Problem Formulation | 1.25–1.5 |
| FALI | 2.75–3.25 |
| Method Validation | 1.25–1.5 |
| GPT-2 IOI | 2.5–2.75 |
| Pythia induction | 1.5–1.75 |
| Discussion | 1.0–1.25 |
| Limitations | 0.75–1.0 |
| Conclusion + declarations | 0.6–0.75 |
| References | 3–4 |
| **Expected total** | **~19–21** |

## Material reserved for Supplementary Information

Move out of the main narrative unless needed to resolve a reviewer-critical issue:

- full Owen derivation and all 36 admissible-permutation details;
- complete 32-cell intervention table;
- every EF-1 validation threshold/error;
- full per-seed/per-template/per-stratum tables;
- template-overlap sensitivity details;
- full candidate-head taxonomy and secondary-head results;
- complete EF-5 scheme diagnostics and damage-strength tables;
- implementation/environment/hardware details beyond reproducibility essentials;
- extra descriptive FALI/QK diagnostics;
- full preregistration chronology and code-audit history.

## Writing constraints

- Dense, plain scientific prose; no rhetorical filler.
- Prefer one claim per sentence when evidence differs.
- Numerical claims must carry uncertainty where available.
- Distinguish preregistered confirmation from exploratory/descriptive results.
- Do not call a positive conditional gain 'recruitment' unless conditional effect is also positive under the frozen functional definition.
- Never imply that FALI cells are naturally occurring states or unique causal pathways.
