# Phase-1B Review Summary

**Date (UTC):** 2026-08-13T19:29:26.336135+00:00  
**Branch:** `revision/fali-validity`  
**Stop condition:** Full Phase-1 complete. No FALI Phase 2.

## 1. Modal GPU

- GPU: **NVIDIA L4** (`L4` requested)
- torch: 2.6.0+cu124
- CUDA: 12.4
- transformer-lens: 3.2.1
- transformers: 4.57.6
- model/token devices: `cuda:0` / `cuda:0`
- dtype policy: FP32, no mixed precision

## 2–3. CUDA parity

Compared `results/revision_phase1/smoke/` (CPU) vs `results/revision_phase1/modal_smoke_seed42/` (CUDA), per-example for all IO−S and attention quantities across 12 candidates × 8 examples.

- max abs IO−S error: **4.29e-05** (threshold 1e-3) → **PASS**
- max abs attention error: **1.58e-06** (threshold 1e-4) → **PASS**
- overall: **PASS** → full N=200 runs authorized and executed

## 4. Four-seed primary-ablation table (mean IO−S)

| condition | mean across seeds | SD | seed42 | seed1 | seed8 | seed22 |
|---|---:|---:|---:|---:|---:|---:|
| A_clean | 2.9682 | 0.0798 | 2.9732 | 2.8741 | 3.0687 | 2.9567 |
| B_old_historical | 2.0847 | 0.0678 | 2.1055 | 1.9958 | 2.0794 | 2.1583 |
| B_corrected | 2.8007 | 0.0362 | 2.8328 | 2.7487 | 2.8081 | 2.8130 |
| L9H6_only | 3.3986 | 0.1092 | 3.4154 | 3.3078 | 3.5464 | 3.3250 |
| L9H9_only | 2.4814 | 0.0817 | 2.4357 | 2.3938 | 2.5741 | 2.5223 |
| L10H0_only | 2.7214 | 0.0699 | 2.7617 | 2.6183 | 2.7672 | 2.7382 |

### Primary-drop comparison

| seed | A | B_old | B | hist drop A−B_old | corr drop A−B | difference (corr−hist) |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 2.9732 | 2.1055 | 2.8328 | 0.8677 | 0.1404 | -0.7273 |
| 1 | 2.8741 | 1.9958 | 2.7487 | 0.8784 | 0.1254 | -0.7530 |
| 8 | 3.0687 | 2.0794 | 2.8081 | 0.9893 | 0.2606 | -0.7287 |
| 22 | 2.9567 | 2.1583 | 2.8130 | 0.7984 | 0.1437 | -0.6547 |

**Note:** Corrected knockout (adding L9H6) produces a **smaller** IO−S drop than historical B_old on every seed. L9H6-only ablation *increases* mean IO−S above clean (negative degradation). Do not assume corrected knockout is a stronger knockout.

## 5. Top conditional-gain heads

```
 rank head_name     head_role  mean_conditional_gain_across_seeds  mean_attention_spike_across_seeds                       recruitment_label
    1     L10H7      negative                            1.479033                          -0.435124 strong_positive_conditional_recruitment
    2     L10H2        backup                            0.200339                           0.304732 strong_positive_conditional_recruitment
    3     L11H2        backup                            0.114720                           0.156501 strong_positive_conditional_recruitment
    4      L9H7        backup                            0.072038                           0.000000 strong_positive_conditional_recruitment
    5     L10H1        backup                            0.070233                           0.300561 strong_positive_conditional_recruitment
    6     L10H3 extra_control                            0.037740                           0.177502 strong_positive_conditional_recruitment
    7     L10H5 extra_control                            0.009492                          -0.001649 strong_positive_conditional_recruitment
    8      L9H0        backup                            0.007137                           0.000000         suggestive_positive_recruitment
```

## 6. L10H10 four-seed result

- Mean conditional gain: **-0.0862**
- Attention spike remains large (~0.32)
- Classification: **`stable_negative_conditional_gain`**
- Positive gain seeds: **0/4**; negative CIs excluding zero: **4/4**
- Descriptive: L10H10 does **not** show stable positive conditional functional recruitment under corrected primary knockout.

## 7. L10H2 four-seed result

- Mean conditional gain: **0.2003**
- Classification: **`strong_positive_conditional_recruitment`**
- Positive gain seeds: **4/4**; positive CIs excluding zero: **4/4**

## 8. Documented backups with strong positive recruitment

L9H7, L10H1, L10H2, L11H2

Full strong-positive set (all roles): L9H7, L10H1, L10H2, L11H2, L10H7, L10H3, L10H5

## 9. Attention spike vs conditional gain correlation

```
           subset  n  pearson_r  pearson_p  spearman_r  spearman_p
all_12_candidates 12  -0.752052   0.004782   -0.217163    0.497785
backup_heads_only  8   0.075712   0.858581   -0.095810    0.821452
```

Across all 12 candidates Pearson is strongly negative (effect size), largely because large-spike / negative-gain L10H10 and large positive-gain / negative-spike L10H7. Among **backup heads only**, correlation is near zero (Pearson ≈ 0.08). Do not treat attention increase as recruitment.

## 10. Unexpected findings

1. **L10H10 fails the recruitment criteria** despite a large corrected attention spike.
2. **Corrected primary knockout is weaker** (smaller IO−S drop) than historical L9H9+L10H0 because L9H6 behaves anti-helpful on this metric when ablated alone.
3. **L10H7** (canonical Negative Name Mover) has the largest positive conditional gain — role-consistent suppressor dynamics, not “backup copy” behavior.
4. Several backups (**L10H2, L11H2, L9H7, L10H1**) show strong positive recruitment; **L10H6** does not.

## 11. Historical integrity

Headline historical SHA256 hashes unchanged vs pre-Phase-1B snapshot.

## 12. ZIP path

`/Users/ronny/Desktop/Research/AI ALIGNMENT/FALI/phase1b_review_package.zip`
