# Manuscript Positioning — Frozen Baseline

Status: FROZEN for manuscript reconstruction unless a new literature audit uncovers a direct precedent.

## Working identity

**FALI = Factorized Attention-Level Intervention.**

Preferred title:

> **FALI: Mechanistic Decomposition of Conditional Attention-Head Importance**

## Central research question

> When an attention head's causal usefulness changes after another circuit component is perturbed, which changes in and around that head account for the shift?

## Core positioning

FALI is **not** a circuit-discovery or backup-head detection method. It is a **mechanism-discrimination framework for already-identified conditionally important attention heads**.

Given intact state A and perturbed state B, FALI decomposes the cross-state change in the candidate head's output-grounded causal usefulness under an explicit counterfactual intervention game into:

- **T** — target attention score(s),
- **R** — rival/competing scores in the task-relevant query row,
- **O** — other structurally valid attention-score coordinates,
- **V** — candidate value content,
- **C** — surrounding-network/context state.

The standard Owen value is applied to the frozen coalition structure SCORE={T,R,O}, VALUE={V}, CONTEXT={C}. The mathematical attribution method itself is not claimed as novel.

## Primary novelty claim

Preferred wording:

> **To our knowledge, FALI is the first framework to decompose the intervention-induced change in an attention head's output-grounded causal usefulness into architecture-aligned score, value, and surrounding-context contributions, while resolving the score contribution into target, rival, and other-score channels.**

This sentence must remain qualified with **"to our knowledge"** and must be immediately contextualized against the closest prior work.

## Intellectual location relative to prior work

The manuscript should present the field as a three-stage progression:

1. **Localization / self-repair / conditional importance** — identifies that a component matters, or that its importance changes after another component is removed.
2. **Interaction-aware causal attribution** — establishes that component effects depend on the state of the rest of the model.
3. **FALI mechanism resolution** — asks which architecture-level changes account for that conditional change for a selected attention head.

Compact framing:

> Localization tells us **where**; conditional attribution tells us **when importance changes**; FALI asks **why that change occurs under a specified intervention model**.

## Closest prior work and boundary

### Self-repair / backup mechanisms

Hydra, Copy Suppression, and broader self-repair work already establish that compensation after ablation exists and can arise through multiple mechanisms. Therefore we do **not** claim first discovery of self-repair, backup heads, or heterogeneous repair.

Key references:
- *The Hydra Effect: Emergent Self-repair in Language Model Computations* — arXiv:2307.15771.
- *Copy Suppression: Comprehensively Understanding an Attention Head* — arXiv:2310.04625.
- *Explorations of Self-Repair in Language Models* — arXiv:2402.15390.

### Conditional importance

CoAx explicitly formalizes conditional component importance and demonstrates backup recovery in IOI and induction. Therefore FALI does **not** claim to invent conditional recruitment or conditional co-ablation.

Key reference:
- *Conditional Co-Ablation: Recovering Self-Repair Backups in Transformer Circuits* — arXiv:2607.01940.

### Interaction / context dependence

Recent multi-mediator work establishes that causal effects depend on the surrounding model state and that interaction can strongly change component rankings. FALI must not claim first discovery of context-dependent causal effects.

Key reference:
- *The Curse of Multiple Mediators* — arXiv:2606.27510.

### Attention routing versus content

QK/OV analyses, causal attention methods, and recent QK-versus-value decompositions already separate routing from transported content. Therefore FALI's novelty is **not** merely separating attention pattern/score computation from values.

Key references:
- *A Mathematical Framework for Transformer Circuits* — Transformer Circuits, 2021.
- *Causal Head Gating* — arXiv:2505.13737.
- recent causal QK/value decomposition of function vectors — arXiv:2605.16591.

### Attribution mathematics

Shapley/Owen attribution is established mathematics. FALI contributes the intervention game and mechanistic factorization, not the value-allocation theorem itself.

## Three manuscript contributions

1. **Methodological.** Define an output-grounded counterfactual intervention game for explaining the change in a selected attention head's causal usefulness between intact and perturbed circuit states.
2. **Mechanistic.** Resolve that change into score computation, value content, and surrounding-network context; within scores, resolve target, rival, and other-score channels while preserving coarse/fine consistency through standard Owen attribution.
3. **Empirical.** Show that similar conditional recruitment can conceal different fine-grained mechanisms: GPT-2 IOI and Pythia induction both exhibit recruitment, but their target/rival score contributions have opposite signs.

## Frozen empirical headline

> **Conditional recruitment generalizes; its fine mechanism does not.**

Project evidence:
- GPT-2 canonical confirmation: `results/extended_fali/ef4_canonical/aggregate/`
- Ablation robustness: `results/extended_fali/ef5_ablation_robustness/`
- Pythia induction transfer: `results/extended_fali/ef6_pythia_induction/aggregate/`

The Pythia failure of the preregistered GPT-2 sign-generalization gate must be presented as a negative result, not retroactively relabeled as a successful generalization hypothesis.

## Claims explicitly prohibited

Do not claim:

- first discovery of self-repair or backup heads;
- first discovery of conditional component importance;
- first demonstration that head roles depend on other heads/context;
- first demonstration of multiple self-repair mechanisms;
- first QK/OV or routing/value decomposition;
- first factorial intervention on attention/content;
- mathematical novelty of Shapley or Owen values;
- that C is a uniquely identified downstream mechanism;
- that hybrid FALI states are naturally occurring model states;
- that FALI recovers the unique natural causal pathway;
- a universal target/rival recruitment mechanism.

## Required limitations language

The final paper must explicitly state:

- FALI is hypothesis-driven and requires a selected candidate head and task-relevant target edge/set.
- Hybrid interventions can be off-manifold; FALI is an explicit counterfactual decomposition, not an ontological claim about unique natural pathways.
- C is a coarse surrounding-network/context contribution, potentially aggregating changes in other heads, MLPs, residual representations, normalization, and upstream/downstream computation.
- Attribution depends on the frozen coalition structure and intervention semantics.
- Current substantive demonstrations cover two model/task settings, not universal transformer behavior.

## Change-control rule

Any sentence asserting novelty, precedence, broad generalization, or causal interpretation must appear in `docs/CLAIM_CITATION_MATRIX.md` before entering the manuscript. If a new paper weakens a claim, update the matrix and narrow the manuscript claim before submission.
