# Machine Learning (Springer) scientific adaptation

## Research-lead status

The scientific content is frozen. This venue conversion must not change experimental results, causal estimands, novelty boundaries, failed preregistered outcomes, or limitations. The current Neural Networks production manuscript is the scientific source of truth; this document specifies the limited Machine Learning-specific editorial adaptation.

## Recommended title

**FALI: Mechanistic Decomposition of Conditional Attention-Head Causal Usefulness**

Keep the title unchanged. It is concise, method-centered, and accurately names the estimand.

## Machine Learning positioning

Machine Learning explicitly accepts research-methodology papers that improve how ML research is conducted, provided claims are supported by verifiable or replicable evidence. Position FALI as a **mechanistic-interpretability research methodology** for explaining state-dependent causal usefulness in learned neural networks. Do not frame it as a new self-repair theory, new circuit-discovery algorithm, or new cooperative-game theorem.

The paper's ML-facing question is:

> When an internal component of a learned neural network becomes more or less causally useful after another component is perturbed, how can we distinguish whether the change arises from the component's routing scores, its transported content, or the altered surrounding network?

## Recommended abstract (venue-adapted; 150-250 words)

Mechanistic analyses of learned neural networks often compare component importance before and after an intervention, but detecting conditional importance does not explain why the component's causal contribution changed. We introduce FALI (Factorized Attention-Level Intervention), an output-grounded counterfactual methodology that decomposes the cross-state change in a selected attention head's causal usefulness into target, rival, and other attention-score contributions, value content, and surrounding-network context. Standard Owen attribution allocates interactions while reconciling the fine score decomposition with a coarse score/value/context game. Controlled synthetic systems and a randomly initialized transformer validate factor isolation, endpoint recovery, intervention isolation, and coarse--fine consistency. In GPT-2 Small on indirect object identification, L10H2 is positively recruited after primary Name Movers are ablated and exhibits a target-positive/rival-negative score decomposition, whereas L10H10 shows stronger target-attention adaptation without positive recruitment. A prospectively specified transfer test in Pythia-160M induction again finds positive recruitment, but the target and rival contributions reverse sign. Thus, the same high-level functional response can arise through different fine-grained mechanisms. FALI provides a verifiable mechanism-discrimination stage for selected conditionally important attention heads under explicit intervention semantics.

## Recommended keywords

- mechanistic interpretability
- causal intervention
- transformer circuits
- attention heads
- conditional importance
- causal attribution

## Replacement Introduction

Machine-learning models can redistribute causal responsibility when their internal computation is perturbed. Ablating one component can change the contribution of others, producing self-repair, backup behavior, or more general conditional importance. For mechanistic interpretation, this creates a methodological problem: detecting that a component became more or less important does not identify why its contribution changed. A selected attention head may alter its task-relevant routing scores, its competing scores, or its value content; alternatively, essentially the same candidate computation may become more or less useful because the surrounding network has changed. Reliable mechanistic evidence therefore requires separating the detection of conditional importance from the explanation of its mechanism.

We introduce **FALI (Factorized Attention-Level Intervention)**, a hypothesis-driven methodology for resolving this ambiguity. FALI begins after a candidate attention head and a task-relevant target edge have been identified. It compares an intact state with a prespecified perturbed state and decomposes the resulting change in the candidate's *output-grounded causal usefulness*. The candidate's attention computation is factorized into target scores $T$, rival scores $R$, other structurally valid scores $O$, value content $V$, and surrounding-network context $C$. Evaluating the resulting $2^5$ counterfactual states defines a causal game whose interactions are allocated with the standard Owen value. The fine score contributions reconcile exactly with a coarser SCORE/VALUE/CONTEXT decomposition.

The key distinction is between *localizing conditional importance* and *explaining it*. Activation and path patching identify behaviorally relevant components or paths; conditional co-ablation identifies components whose importance changes after a primary intervention; interaction-aware mediation analyses show that a component's measured effect can depend strongly on the rest of the model. FALI is downstream of these approaches: given a selected conditional change, it asks which architecture-aligned changes account for it under an explicit intervention model. This makes FALI a mechanism-resolution methodology rather than an automatic circuit-discovery procedure.

We first validate the intervention semantics in controlled systems. Deterministic synthetic attention models isolate each of $T,R,O,V,C$ and a known cross-factor interaction, while a small randomly initialized transformer checks score-to-pattern reconstruction, value-to-output reconstruction, endpoint recovery, intervention isolation, batch-specific score partitions, and consistency between the fine Owen game and its coarse quotient game. These tests establish that the implementation recovers the factors it manipulates; they do not imply that the factorization is unique or that mixed hybrid states are naturally occurring model states.

We then apply FALI to GPT-2 Small on indirect object identification (IOI). Corrected functional analysis separates attention adaptation from recruitment: after the canonical primary Name Movers are removed, L10H2 becomes more causally useful, whereas L10H10 increases target attention even more strongly while its causal usefulness decreases. FALI assigns both heads a target-positive/rival-negative score pattern, showing that even a favorable-looking score reorganization does not itself define recruitment. For positively recruited L10H2, the main fine-score contributions are $\phi_T>0$ and $\phi_R<0$ across the canonical confirmation cohort.

The decisive test is cross-setting transfer. We prospectively test the GPT-2 target-positive/rival-negative mechanism in Pythia-160M induction. Positive conditional recruitment appears again, but the fine score signs reverse: $\phi_T<0$ and $\phi_R>0$. The mechanism-transfer hypothesis therefore fails. The shared functional label---conditional recruitment---does not imply a shared fine-grained mechanism.

Our contributions are threefold. First, we define an output-grounded counterfactual estimand for the cross-state change in a selected attention head's causal usefulness. Second, we factor that change into score, value, and surrounding-context contributions, with target/rival/other resolution inside SCORE and coarse--fine consistency via standard Owen attribution. Third, we show empirically that two settings with positive conditional recruitment can exhibit opposite target/rival score mechanisms. The contribution is therefore a research methodology for discriminating mechanisms of conditional importance in learned neural networks, not a new theory of self-repair, a new localization method, or new Shapley/Owen mathematics.

An earlier public preprint from this project used the name FALI for a narrower **Frozen Attention Logit Intervention** analysis of attention-probability changes in GPT-2 IOI. The present paper substantially revises and supersedes that formulation: it changes the estimand to output-grounded causal usefulness, expands the intervention game to $T/R/O/V/C$, adds controlled implementation validation and coarse--fine reconciliation, broadens the GPT-2 confirmation, and adds a prospectively specified cross-setting mechanism-transfer test. The detailed relationship is stated in Related Work and in the MLJ Contribution Information Sheet.

## Related Work decision

The current Related Work is scientifically suitable for Machine Learning and should be retained with only venue-neutral copyediting. Preserve these boundaries:

- CoAx: identifies which components become conditionally important; FALI asks why a selected attention head's usefulness changed.
- Curse of Multiple Mediators: establishes interaction/context dependence; FALI does not claim priority for context-dependent causal effects.
- Activation/path patching: localization/causal testing rather than the present cross-state factorization estimand.
- QK/OV and routing/content decompositions: established distinctions; not FALI novelty.
- Owen/Shapley attribution: established mathematics; not FALI novelty.
- Earlier Frozen Attention FALI preprint: explicitly superseded, not silently replaced.

## Discussion decision

Retain the current Discussion structure. It already makes the Machine Learning-relevant methodological point: a common functional label can conceal different mechanisms, and FALI should be used downstream of localization. Do not strengthen the cross-setting result into a universal claim.

## LLM disclosure for Springer placement

Springer states that substantive LLM use should be documented in the Methods section (or a suitable alternative). Replace the Elsevier-style standalone declaration with the following paragraph near the end of the Methods section:

> **Use of generative AI tools.** Generative AI tools, including OpenAI ChatGPT and Cursor, were used during manuscript development for literature-search support, drafting and editing, code generation and review, figure-layout development, and LaTeX/repository integration. All experimental designs, analyses, numerical results, citations, scientific claims, and final manuscript content were reviewed and verified by the authors against the underlying source code, result artifacts, and primary literature. The authors retain full responsibility for the work and its conclusions.

## Statements and Declarations

Use Springer-style declarations, not Elsevier front matter:

**Funding.** This research received no external funding.

**Competing interests.** The authors declare that they have no financial or non-financial interests that are directly or indirectly related to the work submitted for publication.

**Data and code availability.** Keep the current evidence-preserving statement until the archival DOI/URL exists. Before submission, replace future-tense language with the actual public repository and persistent archive identifiers if available.

**Author contributions.** Convert the existing CRediT statement into Springer-compatible author-contribution prose without changing contribution claims.

## Scientific safeguards for the conversion

1. L10H2 is positive functional recruitment; L10H10 is not.
2. GPT-2 canonical fine pattern: $T>0$, $R<0$.
3. Pythia positive recruitment recurs, but $T<0$, $R>0$; the preregistered transfer hypothesis failed.
4. EF-5 matched alternatives preserve the tested fine-channel directions, but the overall preregistered robustness gate failed.
5. The gauge sensitivity supports only sign robustness to the prespecified zero-mean row-centering analysis, not full gauge or magnitude invariance.
6. $C$ is a coarse surrounding-network/context contribution, not a localized downstream pathway.
7. Do not claim novelty for self-repair, conditional importance, QK/OV separation, interaction dependence, factorial attention intervention, or Owen/Shapley mathematics.
