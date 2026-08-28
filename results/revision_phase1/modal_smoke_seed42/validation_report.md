# Phase-1 smoke/validation report

## Dataset validation
- All examples passed S×2 / IO×1 / end==' to' checks.
- tokens device=cuda:0; model device=cuda:0

## Primary knockout
- mean A=3.521107, B_old=2.144470, B=2.906307
- Corrected B differs from historical B_old.
- Corrected primary hooks (one per layer): ['blocks.9.attn.hook_z', 'blocks.10.attn.hook_z']

## Candidate execution
- Evaluated 12 candidates (D_h and C_h).
- Per-example identity `conditional_gain = (B-C)-(A-D)` verified.
- Attention values finite and in [0,1] (tol 1e-6).

## Historical result integrity
- Historical headline artifacts unchanged: **True**
  - `results/summary/attention_results.json` sha256=None
  - `results/summary/ablation_controls_summary.json` sha256=None
  - `results/summary/output_mediation_summary.json` sha256=None
  - `results/raw/fali_decomposition_raw.csv` sha256=None

## Smoke means (not for scientific interpretation)
- A mean IO−S: 3.521107
- B_old mean IO−S: 2.144470
- B mean IO−S: 2.906307

## Candidate summary (smoke)
```
head_name     head_role  mean_corrected_attention_spike  mean_intact_effect  mean_conditional_effect  mean_conditional_gain
     L9H0        backup                        0.000000           -0.096337                -0.073029               0.023307
     L9H7        backup                        0.000000            0.339345                 0.404538               0.065193
    L10H1        backup                        0.258775           -0.031719                 0.004196               0.035915
    L10H2        backup                        0.305322            0.148903                 0.499727               0.350824
    L10H6        backup                        0.240462            0.216416                 0.207995              -0.008421
   L10H10        backup                        0.301718            0.266508                 0.335844               0.069336
    L11H2        backup                        0.123921           -0.273879                -0.335454              -0.061575
    L11H9        backup                        0.083455           -0.070076                -0.114473              -0.044397
    L10H7      negative                       -0.409597           -1.289985                -0.284616               1.005369
   L11H10      negative                       -0.020122           -0.490327                -0.458133               0.032194
    L10H3 extra_control                        0.164074            0.069736                 0.067827              -0.001909
    L10H5 extra_control                       -0.002601           -0.033267                -0.040787              -0.007520
```

- Runtime seconds: 15.08
- N=8 smoke only; do not interpret scientifically.
