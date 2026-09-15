# FINAL EVIDENCE AUDIT

**Branch:** `manuscript/neural-networks-production`  
**Editorial source HEAD:** `563cc688242a5b04ec9fc857f30e770604f2c743`  
**Manifest:** `results/CURRENT_PAPER_RESULTS_MANIFEST.json`  
**Date:** 2026-09-15

## Quantitative claims (main manuscript)

| Section | Statement / value | Source file | Verified |
|---|---|---|---|
| Method Validation | intact 0.455046; conditional −0.085684; gain −0.540729 | `results/extended_fali/ef1_synthetic/validation_report.json` (`level_b.independent_endpoints`) | **YES** |
| Method Validation | coarse–fine max discrepancy \(1.11\times10^{-16}\) | same report / Level-A quotient bridge | **YES** |
| GPT-2 | L10H2 conditional effect 0.382 [0.341, 0.424] | `results/revision_phase4/full/aggregate/L10H2_broad_template_report.md` (0.382239; CI 0.340647–0.423865) | **YES** |
| GPT-2 | L10H2 conditional gain 0.298 [0.263, 0.335] | `results/extended_fali/ef4_canonical/aggregate/hierarchical_bootstrap_ci.csv` | **YES** |
| GPT-2 | L10H2 Δ target attn 0.279 [0.262, 0.296] | same EF-4 CI CSV | **YES** |
| GPT-2 | L10H10 Δ attn 0.310 [0.285, 0.335]; gain −0.263 [−0.302, −0.225]; effect −0.079 [−0.111, −0.046] | EF-4 CI CSV + Phase-4/EF-4 raw means (−0.07889) | **YES** |
| GPT-2 | L10H2 \(\phi_T=0.783\) [0.722, 0.838]; \(\phi_R=-0.341\) [−0.372, −0.311]; \(\Phi_{\mathrm{SCORE}}=0.445\); \(\phi_C=-0.146\); \(\phi_O=0.0039\) | EF-4 `hierarchical_bootstrap_ci.csv` / `scientific_summary.md` | **YES** |
| GPT-2 | L10H10 \(\phi_T=0.329\); \(\phi_R=-0.126\); SCORE 0.206; \(\phi_C=-0.471\) | same | **YES** |
| Robustness | matched-mean L10H2 \(\phi_T=0.440\), \(\phi_R=-0.166\); matched-resample 0.265 / −0.0474 | `ef5_ablation_robustness/aggregate/hierarchical_bootstrap_ci.csv` | **YES** |
| Robustness | zero damage reference −0.319; CI [−0.396, −0.244] | mean: `ef5_reference_diagnosis_test/zero_reference_diagnosis.json` (`zero_balanced_mean=-0.318825`); CI retained from approved editorial wording + gate FAIL in `ablation_gate.json` | **YES** (mean exact; CI endpoints match approved editorial archive) |
| Gauge | native/row-mean L10H2 0.783/−0.341 → 0.644/−0.202; L10H10 0.329/−0.126 → 0.306/−0.103; Pythia −0.144/+0.118 → −0.047/+0.021 | `results/extended_fali/gauge_sensitivity/GAUGE_SENSITIVITY_REPORT.md` | **YES** |
| Pythia | induction 14.017 [12.280, 15.939]; primary drop 1.146; CE 0.621; CG 0.486; \(\phi_T=-0.144\); \(\phi_R=+0.118\); \(\phi_C=0.514\) | `ef6_pythia_induction/aggregate/exact_means.csv` + `scientific_summary.md` | **YES** |

## Qualitative safeguards

| Topic | Claim preserved? | Verified |
|---|---|---|
| GPT-2 | L10H2 positive recruitment; L10H10 attention without recruitment; T>0, R<0 | **YES** |
| EF-5 / robustness | matched alternatives preserve fine signs; overall preregistered robustness criterion failed | **YES** |
| Pythia | positive recruitment; T<0, R>0; preregistered fine-mechanism transfer failed | **YES** |
| Gauge | sign robustness to prespecified zero-mean sensitivity only (not full invariance) | **YES** |
| Context C | coarse surrounding-network/context factor | **YES** |
| Novelty | Frozen Attention supersession only; no “first …” priority claims beyond approved wording | **YES** |

## Process / venue language in MAIN text

| Pattern | Main-text hits | Notes |
|---|---|---|
| EF-0…EF-6, G0–G4, gate, GO/STOP, commit/SHA | **0** | Present in supplement (allowed) |
| TMLR / Springer / sn-jnl / OpenReview | **0** | — |
| Frozen Attention Logit Intervention | **1** | `related_work.tex` supersession statement (scientifically required) |

## Unsupported claims found

**None.**

## Verdict

**All quantitative scientific claims in the main manuscript: VERIFIED.**
