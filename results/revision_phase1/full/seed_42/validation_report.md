# Phase-1 smoke/validation report

## Dataset validation
- All examples passed S×2 / IO×1 / end==' to' checks.
- tokens device=cuda:0; model device=cuda:0

## Primary knockout
- mean A=2.973241, B_old=2.105513, B=2.832847
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
- A mean IO−S: 2.973241
- B_old mean IO−S: 2.105513
- B mean IO−S: 2.832847

## Candidate summary (smoke)
```
head_name     head_role  mean_corrected_attention_spike  mean_intact_effect  mean_conditional_effect  mean_conditional_gain
     L9H0        backup                        0.000000           -0.101779                -0.098435               0.003344
     L9H7        backup                        0.000000            0.334425                 0.400537               0.066112
    L10H1        backup                        0.301576            0.008331                 0.089382               0.081051
    L10H2        backup                        0.299988            0.113327                 0.312470               0.199143
    L10H6        backup                        0.262066            0.265571                 0.259212              -0.006360
   L10H10        backup                        0.321194            0.237232                 0.163073              -0.074160
    L11H2        backup                        0.149705           -0.361793                -0.326916               0.034877
    L11H9        backup                        0.101285           -0.069925                -0.101478              -0.031553
    L10H7      negative                       -0.428456           -1.671888                -0.250845               1.421043
   L11H10      negative                        0.009548           -0.785785                -0.865296              -0.079511
    L10H3 extra_control                        0.170897            0.117512                 0.164661               0.047150
    L10H5 extra_control                       -0.001740           -0.032193                -0.018159               0.014034
```

- Runtime seconds: 13.77
- N=8 smoke only; do not interpret scientifically.
