Step 1: Stabilize the repo and separate the interventions
Goal: make the project reproducible and stop mixing different counterfactuals.
Changes:
Add requirements.txt or environment.yml.
Add one canonical runner script, e.g. run_all_experiments.py.
Add a clear folder structure: src/, experiments/, results/raw/, results/summary/, figures/.
Fix the incorrect interpretation in hybrid_intervention.py: FALI and LN-frozen are different interventions.
Rename metrics so they are explicit:
fali_target_only_hybrid
ln_scale_frozen
standard_zero_ablation
Stop calling L10H0 a backup head when it is part of the primary ablation condition.
Save raw per-example results, not just means.
Success criteria:
One command reproduces the current baseline outputs.
FALI and LN-frozen results are saved separately.
No script text claims “definitive proof,” “retire hypothesis,” or “pure active routing.”
Step 2: Add better ablation controls
Goal: test whether the effect survives beyond zero ablation.
Changes:
Keep current zero ablation.
Add mean ablation.
Add resample ablation from other IOI examples.
Use the same dataset, seed, and metric extraction for all ablation types.
Report clean, zero, mean, and resample results side by side.
Success criteria:
L10H10 attention and IO-S logit difference are reported for every ablation type.
Results show whether the self-repair signal is robust or zero-ablation-specific.
Step 3: Implement proper active/passive decomposition
Goal: avoid relying on one path-dependent FALI decomposition.
Changes:
Implement target-first FALI: ablated IO logit, clean non-IO logits.
Implement background-first FALI: clean IO logit, ablated non-IO logits.
Implement Shapley-style two-term decomposition averaging both orderings.
Report active/passive estimates from all three.
Keep LN-scale-freezing as a separate LayerNorm-specific diagnostic, not the main active/passive split.
Success criteria:
The code outputs target-first, background-first, Shapley, and LN-frozen results separately.
The report explicitly states whether the decomposition is stable across methods.
Step 4: Decompose QK score into norm and direction effects
Goal: test whether the attention shift is due to vector direction, vector magnitude, or both.
Changes:
For each Layer 10 head, compute:
q_norm
k_norm
qk_cosine
qk_dot
attention_score = q_dot_k / sqrt(d_head)
Compare clean vs ablated vs LN-frozen.
Add per-example and mean outputs.
Success criteria:
The repo can say how much of the QK score change is norm-driven versus cosine/direction-driven.
Cosine is no longer treated as full proof by itself.
Step 5: Add output-level causal tests
Goal: prove whether changed L10H10 attention actually helps IO prediction.
Changes:
Patch L10H10 attention pattern from clean into ablated run.
Patch L10H10 attention pattern from ablated into clean run.
Patch L10H10 OV output separately.
Measure downstream IO-S logit difference.
Compare L10H10 with control heads.
Success criteria:
The final claim is based on output-level causal effect, not only attention probability.
The repo can distinguish “attention changed” from “attention change helped the model recover.”