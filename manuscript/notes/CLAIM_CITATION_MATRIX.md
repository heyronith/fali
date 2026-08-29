# Claim-to-Citation Matrix

Purpose: prevent novelty inflation, citation gaps, and drift between experimental evidence and manuscript wording. Every novelty, precedence, generalization, and causal-interpretation sentence should be checked against this matrix before entering the final manuscript.

Status labels:
- **GREEN** — supported and safe with stated scope.
- **YELLOW** — defensible only with qualification / explicit closest-work citation.
- **RED** — do not claim.

| ID | Candidate manuscript claim | Status | External literature needed | Internal evidence needed | Safe wording / action |
|---|---|---|---|---|---|
| N1 | FALI is the first method to identify backup heads after circuit damage. | RED | CoAx; Hydra; IOI; self-repair literature | None | Never claim. Backup/self-repair and conditional recovery predate FALI. |
| N2 | Conditional component importance is novel to FALI. | RED | CoAx (arXiv:2607.01940); interaction-aware attribution literature | None | Never claim. FALI takes conditional importance as its explanandum. |
| N3 | Component causal effects can depend on surrounding model state. | RED as novelty | *The Curse of Multiple Mediators* (arXiv:2606.27510); CHG (arXiv:2505.13737) | Project conditional-gain results may illustrate, not establish precedence | State as motivation/background, not novelty. |
| N4 | Separating attention routing from transported value content is novel. | RED | Transformer Circuits QK/OV; recent QK/value causal decomposition (arXiv:2605.16591) | None | Never claim. |
| N5 | Owen/Shapley attribution is mathematically novel. | RED | Standard Owen/Shapley literature | `src/extended_fali_math.py` only validates implementation | Claim only the application/game construction. |
| N6 | FALI decomposes the intervention-induced change in an attention head's output-grounded causal usefulness into score, value, and surrounding-context contributions, with target/rival/other-score resolution inside score. | YELLOW→GREEN if final adversarial search finds no direct precedent | CoAx; *Curse of Multiple Mediators*; QK/value decomposition; attention-causal methods; factorial attention/content work | Formal spec; EF-1 synthetic validation; quotient bridge | Use: “To our knowledge, FALI is the first framework to…” and immediately cite closest neighbors. |
| N7 | FALI discovers conditionally important components. | RED / inaccurate scope | CoAx | Phase-1 identification pipeline | Say FALI is downstream of localization/conditional-ablation: it explains a selected component. |
| N8 | FALI resolves conditionality mechanistically rather than returning another scalar interaction/importance score. | GREEN with qualification | CoAx; *Curse of Multiple Mediators* | T/R/O/V/C formalization + synthetic validation | Good positioning sentence. Avoid implying unique natural pathways. |
| M1 | FALI's five factors sum exactly to the endpoint change in causal usefulness under the specified game. | GREEN | Owen attribution background | EF-0R/EF-1 tests; efficiency identities | State explicitly “under the specified counterfactual intervention game.” |
| M2 | T+R+O equals the coarse SCORE contribution and the quotient SCORE/V/C game reconciles with the historical P/V/C decomposition. | GREEN | Owen quotient property | EF-0R + EF-1 quotient tests; EF-4 historical bridge | Distinguish mathematical identity from empirical equality at matched cells. |
| M3 | C is downstream sensitivity. | RED | None | Current method definition contradicts this | Use “surrounding-network/context contribution.” |
| M4 | C uniquely identifies the external mechanism responsible for recruitment. | RED | Intervention/OOD caveats | Same-layer controls only establish intervention architecture | C is a residual causal channel aggregating surrounding computation. |
| M5 | Hybrid FALI states reveal the unique natural causal pathway used by the model. | RED | Intervention-distribution / off-manifold literature | Synthetic tests establish semantics, not naturalness | Describe as explicit counterfactual/interchange interventions. |
| M6 | FALI is a general automatic circuit-discovery method. | RED | Circuit-discovery literature / MIB | Method requires candidate head + target edge | State hypothesis-driven scope. |
| E1 | Attention increase is not equivalent to positive functional recruitment. | GREEN | Self-repair/attention-faithfulness literature for context | Corrected Phase-1 conditional effects: L10H2 vs L10H10; `results/revision_phase1/full/aggregate_corrected/` | Strong motivating empirical statement; do not generalize beyond studied setting without qualification. |
| E2 | GPT-2 IOI exhibits genuine conditional recruitment. | GREEN | IOI/CoAx context | Corrected Phase-1 and canonical experiments | Specify heads/primary ablation and task. |
| E3 | In canonical GPT-2 IOI, L10H2/L10H10 show positive target and negative rival contributions under FALI. | GREEN | No external support needed for result; cite prior IOI methods for context | EF-4 exact means, CIs, seed/stratum tests, T0-excluded sensitivity | Report as FALI-game attribution, not pure direct effect. |
| E4 | The GPT-2 fine-score result is robust to alternative ablation values. | GREEN if EF-5 outputs support exact wording | Ablation-sensitivity literature | EF-5 aggregate results | Name tested ablations explicitly; avoid “all ablations.” |
| E5 | Pythia induction exhibits genuine conditional recruitment of L8H2 after L5H0 ablation. | GREEN | Induction/CoAx context | EF-6 G0/G1/G2 confirmation | Specify model/task/head pair. |
| E6 | Pythia reproduces the GPT-2 target-positive/rival-negative fine mechanism. | RED / falsified | None | EF-6 gate FAIL | Explicitly report failure. |
| E7 | Pythia induction instead shows target-negative and rival-positive contributions. | GREEN | No external support needed for result | EF-6R: phi_T≈-0.144, phi_R≈+0.118 with directional CIs | Present as reversal relative to GPT-2. |
| E8 | Conditional recruitment generalizes; its fine mechanism does not. | YELLOW/GREEN within studied settings | Literature on prompt/model circuit variability supports plausibility, not result | GPT-2 + Pythia experiments | Preferred headline. Qualify: “across our two substantive settings.” |
| E9 | There is no universal backup-head mechanism. | YELLOW | Heterogeneous self-repair literature | Only two settings | Safer: “our results argue against treating conditional recruitment as evidence for a single universal fine-score mechanism.” |
| P1 | FALI is complementary to CoAx. | GREEN | CoAx | Phase-1 identifies; FALI decomposes selected heads | Explain: CoAx/conditional ablation identifies **which**; FALI asks **why**. |
| P2 | FALI is complementary to activation/path patching. | GREEN | activation/path patching | FALI assumes known component/edge | Say localization and mechanism-resolution address different stages. |
| P3 | FALI is more faithful/better than patching in general. | RED unless directly benchmarked | Patching benchmark literature | No sufficient project benchmark | Do not claim. |
| P4 | FALI's problem is practically relevant because conditionality can substantially alter component rankings/recovery. | GREEN | CoAx; *Curse of Multiple Mediators* | Project results illustrate | Use external evidence to motivate impact. |
| L1 | The method is validated against known synthetic ground truth. | GREEN | InterpBench/MIB as motivation | EF-1 single-channel + interaction tests; tiny transformer tests | Strong methodological validation claim. |
| L2 | FALI is robust to arbitrary intervention choices. | RED | Ablation robustness literature | EF-5 covers finite tested alternatives | State exact tested interventions only. |
| L3 | The results establish transformer-wide universality. | RED | None | Two substantive settings | Explicit limitation. |

## Required citation clusters for manuscript drafting

### Conditional importance / self-repair
- McGrath et al., *The Hydra Effect*, arXiv:2307.15771.
- McDougall et al., *Copy Suppression*, arXiv:2310.04625.
- Rushing & Nanda, *Explorations of Self-Repair in Language Models*, arXiv:2402.15390.
- *Conditional Co-Ablation: Recovering Self-Repair Backups in Transformer Circuits*, arXiv:2607.01940.

### Interaction-aware attribution
- Activation/path patching literature.
- Nam et al., *Causal Head Gating*, arXiv:2505.13737.
- *The Curse of Multiple Mediators*, arXiv:2606.27510.

### Attention mechanism decomposition
- Elhage et al., *A Mathematical Framework for Transformer Circuits*.
- Attention-Causal Communication / causal attention analysis.
- Recent causal Query-Key versus Value function-vector decomposition, arXiv:2605.16591.
- Recent factorial attention-allocation/content-corruption study, arXiv:2608.22140.

### Evaluation / intervention limitations
- *Transformer Circuit Faithfulness Metrics are not Robust*, arXiv:2407.08734.
- InterpBench / MIB for known-ground-truth and benchmark motivation.
- Work on divergent/off-manifold representations from causal interventions, arXiv:2511.04638.

## Internal evidence map

- Corrected conditional recruitment: `results/revision_phase1/full/aggregate_corrected/`
- Mathematical formalization / standard Owen: `docs/EXTENDED_FALI_FORMAL_SPEC.md`, `src/extended_fali_math.py`
- Synthetic ground truth: `results/extended_fali/ef1_synthetic/`
- Canonical GPT-2 fine-score confirmation: `results/extended_fali/ef4_canonical/aggregate/`
- Ablation robustness: `results/extended_fali/ef5_ablation_robustness/`
- Pythia induction transfer: `results/extended_fali/ef6_pythia_induction/aggregate/`

## Drafting rule

Before adding a sentence that says or implies **first**, **novel**, **general**, **causal mechanism**, **explains why**, **robust**, or **generalizes**, assign it a row here. RED claims are prohibited. YELLOW claims require the listed qualification and citations. GREEN claims still require exact internal/external evidence where applicable.
