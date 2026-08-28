# Phase-1 smoke/validation report

## Dataset validation
- All examples passed S×2 / IO×1 / end==' to' checks.

## Primary knockout
- mean A=3.521108, B_old=2.144467, B=2.906314
- Corrected B differs from historical B_old.
- Corrected primary hooks (one per layer): ['blocks.9.attn.hook_z', 'blocks.10.attn.hook_z']

## Candidate execution
- Evaluated 12 candidates (D_h and C_h).
- Per-example identity `conditional_gain = (B-C)-(A-D)` verified.
- Attention values finite and in [0,1] (tol 1e-6).

## Historical result integrity
- Historical headline artifacts unchanged: **True**
  - `results/summary/attention_results.json` sha256=d72c5e1c629748899a738c4291c0f30e5017fca0753a25c57e9b33eb10fe7c02
  - `results/summary/ablation_controls_summary.json` sha256=befe72a0a5461edd7c08964545c59a26bc6d43f8a2a2fdffae19bd50bb2e5749
  - `results/summary/output_mediation_summary.json` sha256=1ab5d62b521e23fd14243a4d936d48b4c9f056bf539e7cf1e9fb5e696f837be0
  - `results/raw/fali_decomposition_raw.csv` sha256=b600d06eed00484d694d13b20ef2260720c51607247540b422981b815f7d88cc

## Smoke means (not for scientific interpretation)
- A mean IO−S: 3.521108
- B_old mean IO−S: 2.144467
- B mean IO−S: 2.906314

## Candidate summary (smoke)
```
head_name     head_role  mean_corrected_attention_spike  mean_intact_effect  mean_conditional_effect  mean_conditional_gain
     L9H0        backup                        0.000000           -0.096343                -0.073027               0.023316
     L9H7        backup                        0.000000            0.339343                 0.404542               0.065199
    L10H1        backup                        0.258776           -0.031717                 0.004205               0.035922
    L10H2        backup                        0.305322            0.148902                 0.499732               0.350830
    L10H6        backup                        0.240462            0.216411                 0.208003              -0.008408
   L10H10        backup                        0.301719            0.266508                 0.335849               0.069342
    L11H2        backup                        0.123921           -0.273879                -0.335449              -0.061570
    L11H9        backup                        0.083455           -0.070080                -0.114467              -0.044387
    L10H7      negative                       -0.409597           -1.289984                -0.284610               1.005375
   L11H10      negative                       -0.020122           -0.490329                -0.458128               0.032201
    L10H3 extra_control                        0.164073            0.069735                 0.067835              -0.001901
    L10H5 extra_control                       -0.002601           -0.033270                -0.040783              -0.007513
```

- Runtime seconds: 8.93
- N=8 smoke only; do not interpret scientifically.
