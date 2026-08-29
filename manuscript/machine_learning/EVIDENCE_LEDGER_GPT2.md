# GPT-2 Manuscript Evidence Anchors

Status: **ACTIVE CLAIM CONTROL — GPT-2 section**

This sub-ledger records the exact internal evidence used in `gpt2_ioi.tex`. It supplements `EVIDENCE_LEDGER.md`; no number below is inferred from prose or reconstructed from memory.

| Claim / quantity | Status | Exact source | Estimand / role | Point estimate and uncertainty |
|---|---:|---|---|---|
| Canonical EF-4 design | GREEN | `results/extended_fali/ef4_canonical/aggregate/analysis_config.json` | confirmatory fine-score analysis; 4 seeds; N=900/seed; 15 templates × BABA/ABBA; 10k hierarchical bootstrap | seeds 42,1,8,22; 3600 prompts total |
| L10H2 canonical conditional effect | GREEN | `results/revision_phase4/full/aggregate/L10H2_broad_template_report.md` | output-grounded effect under primary knockout | 0.382239; CI [0.340647,0.423865] |
| L10H2 canonical conditional gain | GREEN | `results/extended_fali/ef4_canonical/aggregate/hierarchical_bootstrap_ci.csv` | cross-state change in causal usefulness | 0.298451; CI [0.262940,0.335184] |
| L10H2 target-attention change | GREEN | same EF-4 bootstrap file | descriptive target attention probability change | 0.279460; CI [0.261643,0.295510] |
| L10H10 canonical conditional effect | GREEN | `results/revision_phase4/full/aggregate/L10H10_broad_template_report.md` | output-grounded effect under primary knockout | -0.078890; CI [-0.111127,-0.045726] |
| L10H10 canonical conditional gain | GREEN | EF-4 bootstrap file | cross-state change in causal usefulness | -0.263442; CI [-0.301914,-0.224875] |
| L10H10 target-attention change | GREEN | EF-4 bootstrap file | descriptive target attention probability change | 0.310250; CI [0.285276,0.334539] |
| Earlier L10H10 functional classification | GREEN, distribution-bounded | `results/revision_phase1/full/aggregate_corrected/functional_classification.csv` | mixed two-template functional screen | conditional effect +0.130005; gain -0.086154; class `supportive_but_downregulated` |
| L10H2 coarse FALI | GREEN, previously observed coarse quantity | `results/extended_fali/ef4_canonical/aggregate/exact_means.csv`; bootstrap file | SCORE/V/C quotient | SCORE 0.445252; V -0.000349; C -0.146451 |
| L10H2 fine T | GREEN confirmatory | EF-4 `exact_means.csv`; `hierarchical_bootstrap_ci.csv`; `hypothesis_results.json` | Owen target-score contribution | 0.782806; CI [0.721856,0.838338] (bootstrap table) |
| L10H2 fine R | GREEN confirmatory | same | Owen rival-score contribution | -0.341419; CI [-0.371673,-0.310909] |
| L10H2 O | GREEN descriptive only | same | other-score contribution | 0.003864; no directional hypothesis |
| L10H10 coarse FALI | GREEN, previously observed coarse quantity | EF-4 exact/CI | SCORE/V/C quotient | SCORE 0.206251; V 0.001556; C -0.471249 |
| L10H10 fine T | GREEN confirmatory | EF-4 exact/CI/hypothesis files | Owen target-score contribution | 0.328524; CI [0.300205,0.355043] |
| L10H10 fine R | GREEN confirmatory | same | Owen rival-score contribution | -0.125907; CI [-0.138610,-0.113054] |
| L10H10 O | GREEN descriptive only | same | other-score contribution | 0.003634; no directional hypothesis |
| EF-4 directional gate | GREEN | `results/extended_fali/ef4_canonical/aggregate/scientific_summary.md`; `hypothesis_results.json` | preregistered fine-score confirmation | PASS; all four hypotheses 30/30 strata, all four seed means correct direction; strict T0-excluded sensitivity passes |
| L9H7/L9H0 architectural controls | GREEN architectural invariant | EF-4 `hypothesis_results.json` | same-layer controls under L9 full-z primary intervention | max abs T/R/O/V = 0; max context residual 1.11e-16 |
| EF-5 matched-mean channels | GREEN, tested alternative only | `results/extended_fali/ef5_ablation_robustness/aggregate/channel_robustness.json` | fine-channel intervention robustness | L10H2 T 0.4400 [0.4082,0.4699], R -0.1655 [-0.1783,-0.1521]; L10H10 T 0.1879 [0.1726,0.2021], R -0.0816 [-0.0885,-0.0744] |
| EF-5 matched-resample channels | GREEN, tested alternative only | same | fine-channel intervention robustness | L10H2 T 0.2651 [0.2451,0.2853], R -0.0474 [-0.0572,-0.0380]; L10H10 T 0.0858 [0.0776,0.0941], R -0.0150 [-0.0189,-0.0114] |
| EF-5 global gate | GREEN negative result | `results/extended_fali/ef5_ablation_robustness/aggregate/ablation_gate.json`; `scientific_summary.md` | preregistered experiment-level gate | FAIL: zero reference damage-drop CI not entirely >0; observed reference -0.318825, CI [-0.395660,-0.243713] |

## Wording controls

- Do **not** call L10H10 a positive recruit on the broad canonical cohort; its canonical conditional effect and gain are both negative.
- `supportive_but_downregulated` applies to the earlier mixed two-template functional screen, not universally across prompt distributions.
- Coarse P/V/C quantities on EF-4 canonical cohorts were previously observed and are context, not new held-out confirmation.
- The new EF-4 confirmatory outcomes are the fine T/R/O channels; only T and R had directional hypotheses.
- EF-5 supports robustness only for matched-mean and matched-resample schemes tested. The overall EF-5 gate failed and must be reported.
- Never infer a universal target-positive/rival-negative recruitment mechanism from GPT-2; EF-6 directly tests and falsifies transfer of that sign pattern.
