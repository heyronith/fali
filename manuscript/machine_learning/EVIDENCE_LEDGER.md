# Evidence Ledger — Machine Learning Manuscript

Status: **ACTIVE CLAIM CONTROL**

Purpose: every substantive manuscript claim must map to external literature or committed internal evidence. Novelty/precedence claims remain governed by `../notes/CLAIM_CITATION_MATRIX.md`.

Legend:
- **GREEN** — directly supported by committed evidence / established literature.
- **YELLOW** — allowed only with qualification and final adversarial literature check.
- **RED** — prohibited or contradicted.

## Core method claims

| ID | Planned claim | Status | Required evidence/citations | Allowed wording / constraints |
|---|---|---:|---|---|
| M1 | FALI decomposes the cross-state change in a selected attention head's output-grounded causal usefulness into T/R/O/V/C under a specified counterfactual game. | GREEN | `docs/EXTENDED_FALI_FORMAL_SPEC.md`; `src/extended_fali_math.py`; `src/extended_fali_interventions.py`; EF-1 validation | State intervention semantics explicitly. Do not imply natural occurrence of hybrids. |
| M2 | Owen efficiency gives phi_T+phi_R+phi_O+phi_V+phi_C = endpoint change in usefulness. | GREEN | formal spec; EF-0R/EF-1 math tests | Mathematical property under the defined characteristic function. No math novelty claim. |
| M3 | phi_T+phi_R+phi_O equals the SCORE Shapley value in the quotient game; V/C reconcile to the coarse score/value/context game. | GREEN | formal spec; EF-1 quotient bridge; historical P×V×C bridge validations | Call this coarse/fine reconciliation, not a new Shapley theorem. |
| M4 | Original two-player target/rival FALI is a nested special case when O/V/C are functionally inactive and the outcome is target attention probability. | GREEN | EF-0R/EF-1 special-case tests | Keep as historical/nesting clarification, not main empirical claim. |
| M5 | C is a surrounding-network/context contribution. | GREEN | formal specification and intervention definition | C may aggregate other heads, MLPs, residuals, normalization, upstream/downstream computation. Never call it a uniquely localized downstream mechanism. |
| M6 | FALI is a circuit-discovery method. | RED | contradicted by design | Never claim. Candidate head and task-relevant target edge/set are inputs. |
| M7 | Hybrid FALI states are unique natural causal pathways. | RED | intervention semantics; off-manifold literature | Never claim. Call them controlled counterfactual/interchange interventions. |

## Novelty / positioning claims

| ID | Planned claim | Status | Closest prior work | Allowed wording |
|---|---|---:|---|---|
| N1 | FALI is the first backup/self-repair method. | RED | Hydra (arXiv:2307.15771); Copy Suppression (2310.04625); Explorations of Self-Repair (2402.15390); CoAx (2607.01940) | Never claim. |
| N2 | FALI invents conditional component importance. | RED | CoAx (2607.01940); Curse of Multiple Mediators (2606.27510) | Never claim. |
| N3 | FALI is first to show head effects depend on context/other heads. | RED | Causal Head Gating (2505.13737); Curse of Multiple Mediators (2606.27510) | Never claim. |
| N4 | FALI is first to separate attention routing from value/content. | RED | Transformer Circuits QK/OV; causal QK/value decomposition (2605.16591); attention-causal communication | Never claim. |
| N5 | Owen/Shapley mathematics is novel. | RED | established Shapley/Owen literature; recent group-aware attribution (2602.17107) | Never claim. |
| N6 | To our knowledge, FALI is the first framework to decompose intervention-induced change in an attention head's output-grounded causal usefulness into score, value, and surrounding-context contributions while resolving score into target/rival/other channels. | YELLOW | cite CoAx (2607.01940), Curse (2606.27510), QK/value work (2605.16591), QK/OV foundations, Owen prior art | Must retain “to our knowledge”; immediately contextualize closest work; final adversarial search before submission. |
| N7 | FALI is mechanism resolution downstream of localization/conditional attribution. | GREEN | CoAx; Curse; activation/path patching | Safe framing: localization tells where; conditional attribution tells when importance changes; FALI asks why under specified intervention semantics. |

## Corrected functional baseline claims

| ID | Planned claim | Status | Internal evidence | Allowed wording |
|---|---|---:|---|---|
| F1 | Attention increase is not equivalent to positive functional recruitment. | GREEN | `results/revision_phase1/full/aggregate_corrected/functional_classification.csv`; EF-4 exact means | Use L10H2/L10H10 contrast. Positive conditional gain alone is also insufficient if conditional effect remains negative. |
| F2 | L10H2 is a genuine recruited GPT-2 IOI backup under the corrected canonical primary knockout. | GREEN | corrected Phase-1 classification; `results/revision_phase4/full/aggregate/L10H2_broad_template_report.md` (conditional effect); EF-4 conditional gain | Report conditional effect/gain, not attention alone. |
| F3 | L10H10 shows strong target-attention adaptation but becomes less useful under the corrected primary knockout. | GREEN | `results/revision_phase4/full/aggregate/L10H10_broad_template_report.md` (negative broad-canonical conditional effect); EF-4: conditional_gain=-0.2634, CI [-0.3019,-0.2249]; delta_target_attn=+0.3103, CI [0.2853,0.3345] | “Attention adaptation without functional recruitment” / “supportive but downregulated” where consistent with functional taxonomy. |

## GPT-2 canonical claims

| ID | Planned claim | Status | Internal evidence | Exact anchors / constraints |
|---|---|---:|---|---|
| G1 | L10H2 has positive conditional recruitment on canonical GPT-2 IOI cohorts. | GREEN | `results/extended_fali/ef4_canonical/aggregate/exact_means.csv`; `hierarchical_bootstrap_ci.csv` | conditional gain 0.29845; 95% CI [0.26294,0.33518]. |
| G2 | L10H2 target contribution is positive. | GREEN | EF-4 exact/CI; fine-score gate PASS | phi_T=0.78281; CI [0.72186,0.83834]. |
| G3 | L10H2 rival contribution is negative. | GREEN | EF-4 exact/CI; fine-score gate PASS | phi_R=-0.34142; CI [-0.37167,-0.31091]. |
| G4 | L10H10 target contribution is positive and rival contribution negative in canonical EF-4. | GREEN | EF-4 exact/CI; fine-score gate PASS | phi_T=0.32852 [0.30020,0.35504]; phi_R=-0.12591 [-0.13861,-0.11305]. Keep distinct from recruitment: its total conditional gain is negative. |
| G5 | EF-4 preregistered fine-score gate passed for both L10H2 and L10H10. | GREEN | `results/extended_fali/ef4_canonical/aggregate/fine_score_gate.json` | Describe exactly what gate tested. Do not inflate into universal mechanism claim. |
| G6 | Same-layer L9 architectural controls have zero score/value attribution under this intervention family. | GREEN | EF-4 exact means / validation | Architectural invariant under same-layer primary z ablation; not evidence that all same-layer heads are backups. |

## Intervention-robustness claims

| ID | Planned claim | Status | Internal evidence | Allowed wording |
|---|---|---:|---|---|
| R1 | The GPT-2 fine-channel result is robust to arbitrary ablation choices. | RED | EF-5 overall gate failed | Never claim. |
| R2 | The matched-mean and matched-resample EF-5 schemes were each classified ROBUST for the preregistered fine-channel analysis. | GREEN | `results/extended_fali/ef5_ablation_robustness/aggregate/ablation_gate.json`; `scientific_summary.md`; `channel_robustness.json` | Must mention these were the tested matched alternative schemes. |
| R3 | EF-5 as a whole passed. | RED | `EF5_ABLATION_GATE: FAIL` | Never claim. |
| R4 | The preregistered EF-5 overall gate failed because the zero-damage reference perturbation criterion was not satisfied. | GREEN | `ablation_gate.json`; `scientific_summary.md` | Include when discussing robustness so selective reporting is impossible. |

## Pythia induction claims

| ID | Planned claim | Status | Internal evidence | Exact anchors / constraints |
|---|---|---:|---|---|
| P1 | Pythia-160M induction shows positive conditional recruitment of L8H2 after L5H0 perturbation. | GREEN | `results/extended_fali/ef6_pythia_induction/aggregate/exact_means.csv` | conditional effect=0.62120 [0.45591,0.86049]; conditional gain=0.48565 [0.32649,0.72182]. |
| P2 | Pythia target contribution is negative. | GREEN | EF-6R exact means | phi_T=-0.14374; CI [-0.24744,-0.08408]. |
| P3 | Pythia rival contribution is positive. | GREEN | EF-6R exact means | phi_R=0.11780; CI [0.05822,0.21913]. |
| P4 | GPT-2 sign mechanism generalized to Pythia. | RED | EF6_GATE failed; signs reversed | Never claim. |
| P5 | The preregistered mechanism-transfer hypothesis failed. | GREEN | `results/extended_fali/ef6_pythia_induction/aggregate/ef6_gate.json`; EF-6R project record | Present as negative result. |

## Cross-setting synthesis claims

| ID | Planned claim | Status | Evidence | Allowed wording |
|---|---|---:|---|---|
| X1 | Conditional recruitment occurs in both studied model/task settings. | GREEN | GPT-2 L10H2 G1 + Pythia P1 | Always bound to “both studied settings”; do not claim transformer-wide prevalence from these two experiments. |
| X2 | The fine target/rival signs are opposite between the GPT-2 exemplar and Pythia induction. | GREEN | G2/G3 + P2/P3 | Strong central empirical result. |
| X3 | “Conditional recruitment generalizes; its fine mechanism does not.” | YELLOW | X1/X2 + failed EF-6 transfer gate | Use only with immediate scope qualifier: across the two studied settings. Prefer manuscript wording: “Conditional recruitment occurs in both studied settings, while the preregistered fine-score mechanism does not transfer.” |
| X4 | There is no universal recruitment mechanism. | RED/YELLOW | only two substantive settings | Do not state as universal negative. Safer: results argue against treating recruitment itself as evidence for a single shared fine-score mechanism. |
| X5 | A common functional response does not imply a common internal mechanism. | GREEN within studied cases | X1/X2 | Present as the demonstrated lesson of these cases, not a theorem about all models. |

## Related-work factual claims requiring external citations

| ID | Claim | Primary citation targets |
|---|---|---|
| L1 | Self-repair/backup compensation is established prior work. | Hydra 2307.15771; Copy Suppression 2310.04625; Explorations of Self-Repair 2402.15390 |
| L2 | Conditional co-ablation can reveal components missed by first-order importance. | CoAx 2607.01940 |
| L3 | Component causal effects can depend strongly on the surrounding model / multiple mediators. | Curse of Multiple Mediators 2606.27510 |
| L4 | Head roles/interactions can be facilitating/interfering/conditional. | Causal Head Gating 2505.13737 |
| L5 | QK routing and OV/value content are established distinct circuit objects. | Mathematical Framework for Transformer Circuits; relevant causal attention papers |
| L6 | Recent work causally decomposes QK routing vs value updates. | arXiv:2605.16591 |
| L7 | Activation/path patching localizes causal components/paths rather than solving our exact cross-state factorization. | Path Patching / activation patching primary sources |
| L8 | Circuit conclusions can be intervention/faithfulness sensitive. | Transformer Circuit Faithfulness Metrics Are Not Robust 2407.08734 |
| L9 | Prompt-specific circuit heterogeneity exists across transformer settings. | arXiv:2602.13483 |
| L10 | Controlled interventions can produce off-manifold/divergent internal representations. | arXiv:2511.04638 |
| L11 | Ground-truth interpretability benchmarks motivate synthetic/mechanical validation. | InterpBench; MIB arXiv:2504.13151 |

## Prior-version disclosure

| ID | Claim | Status | Evidence |
|---|---|---:|---|
| D1 | Earlier public version used “Frozen Attention Logit Intervention” and centered an attention-based GPT-2 backup interpretation. | GREEN | prior manuscript/preprint; Zenodo May 2026 record; historical repository |
| D2 | Corrected functional analysis invalidated attention increase as the recruitment criterion and motivated the output-grounded FALI reconstruction. | GREEN | corrected Phase-1 results and project audit history |
| D3 | New manuscript is a substantial scientific reconstruction rather than a cosmetic extension. | GREEN | new formal method, EF-1 validation, EF-4 confirmation, EF-5 checks, EF-6 cross-setting test | State transparently in MLJ contribution sheet and manuscript where appropriate. |

## Drafting rule

Before adding any new empirical number to `main.tex`, record:
1. exact result path;
2. estimand definition;
3. point estimate;
4. uncertainty interval / seed structure if available;
5. whether confirmatory, exploratory, descriptive, or architectural-control evidence.

Before adding any new novelty/precedence sentence, update both this ledger and `../notes/CLAIM_CITATION_MATRIX.md`.
