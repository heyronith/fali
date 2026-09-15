# Neural Networks editorial reconstruction report

Branch: `manuscript/neural-networks-editorial-rewrite`
Base: `maintenance/current-fali-cleanup` @ `6d6768a4e437b488e2de2122a12d1cd15488ecef`

## Editorial objective

Transform the scientifically correct migration draft into a journal narrative without changing the evidence, experimental outcomes, or novelty boundaries.

## New narrative spine

1. Conditional importance identifies a functional change but does not explain mechanism.
2. FALI decomposes the change in a selected attention head's output-grounded causal usefulness.
3. Controlled systems validate intervention semantics.
4. GPT-2 IOI separates visible attention adaptation from positive recruitment.
5. Recruited L10H2 shows target-positive/rival-negative fine-score attribution.
6. L10H10 shows that the same score pattern does not itself define recruitment.
7. Pythia induction exhibits positive recruitment but reverses the target/rival signs.
8. Therefore the preregistered GPT-2 fine mechanism does not transfer to the tested second setting.

## Main-text changes

- Sharpened title around causal usefulness.
- Rewrote abstract to foreground the estimand and cross-setting reversal.
- Rebuilt Introduction around problem → gap → FALI → validation → GPT-2 → Pythia → contributions.
- Condensed Related Work and moved defensive novelty discussion out of the Introduction.
- Streamlined the conditional-usefulness formulation.
- Shortened the main FALI method while retaining the T/R/O/V/C construction, utility, endpoint identity, Owen coalition structure, efficiency, and coarse–fine identity.
- Moved full Owen/permutation detail to the existing supplement.
- Compressed method validation; exact controlled values remain in the supplement.
- Reorganized GPT-2 around two scientific questions: whether attention increase implies recruitment, and what changes account for the response.
- Removed EF-stage/gate terminology from the GPT-2 main narrative.
- Reframed ablation robustness as tested-alternative channel robustness while preserving the failed overall preregistered criterion.
- Removed Git commit hashes from the main gauge-sensitivity narrative; provenance remains in the supplement/repository.
- Reorganized Pythia around the transfer question and central failed mechanism-transfer result.
- Removed G0–G4 / EF-6 gate terminology from the main narrative.
- Added the important cross-setting confound limitation: model, task, head pair, perturbation, and metric all change.
- Rewrote Discussion around mechanism discrimination rather than experiment chronology.
- Added explicit score-gauge limitation.
- Tightened Conclusion.
- Reworded Highlights to avoid an overly broad claim that recruitment “generalizes”; now states that positive recruitment recurs while the tested fine mechanism does not transfer.

## Scientific claims deliberately preserved

- Positive conditional gain alone is not sufficient for positive recruitment; perturbed-state effect must also be positive.
- GPT-2 L10H2: positive recruitment, `T>0`, `R<0`.
- GPT-2 L10H10: strong attention adaptation and `T>0`, `R<0`, but negative conditional gain.
- EF-5 matched mean/resample alternatives preserve fine-channel signs; the broader preregistered robustness criterion fails because the zero-ablation reference criterion fails.
- Pythia L8H2: positive recruitment, `T<0`, `R>0`; preregistered fine-mechanism transfer fails.
- Gauge sensitivity preserves the fine-channel signs under prespecified zero-mean row-centering but changes magnitudes, especially in Pythia.
- `C` remains a coarse surrounding-network factor, not a localized mechanism.
- Standard Owen attribution is established mathematics; no mathematical novelty claim.
- FALI is mechanism resolution for selected heads, not automatic circuit discovery.
- Hybrid interventions are explicit counterfactuals and may be off-manifold.

## Detail intentionally retained in Supplementary Information

- Standard Owen 36-permutation implementation details.
- Validation chronology and superseded implementation corrections.
- Exact synthetic single-factor values.
- Random-transformer architecture and exact invariant values.
- Full canonical GPT-2 sampling/bootstrap details and exact tables.
- Internal EF-stage/gate nomenclature where useful for provenance.
- Full alternative-ablation estimates and failed experiment-level gate evidence.
- Full Pythia prospective design and gate bookkeeping.
- Gauge preregistration/execution commit hashes and paired sensitivity tables.

## Production tasks still required locally

These are implementation/production tasks, not scientific rewriting:

1. Compile main manuscript and supplement from this branch.
2. Update the supplement title to match the revised main title.
3. Check cross-references after the section-title changes.
4. Report page count, abstract word count, and unresolved warnings.
5. Fix overfull boxes without negative-spacing hacks or class modifications.
6. Re-render eligible scientific plots/diagrams as vector PDF where the existing deterministic plotting source supports it; do not redraw scientific content by hand.
7. Check figure float placement and caption readability.
8. Verify all empirical numbers against `MIGRATION_EVIDENCE_AUDIT.md` / current result manifests after compilation.
9. Do not reintroduce EF/GATE/commit-hash language into the main prose.
10. Do not rerun model experiments.

## Editorial safety constraints

- No deletion or suppression of failed preregistered outcomes.
- No strengthening from “tested second setting” to universal mechanism claims.
- No claim of full gauge invariance.
- No claim that FALI identifies unique natural pathways.
- No new numerical value without source-backed verification.
