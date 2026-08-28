# Phase-1 smoke/validation report

## Dataset validation
- All examples passed S×2 / IO×1 / end==' to' checks.
- tokens device=cuda:0; model device=cuda:0

## Primary knockout
- mean A=3.068717, B_old=2.079413, B=2.808078
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
- A mean IO−S: 3.068717
- B_old mean IO−S: 2.079413
- B mean IO−S: 2.808078

## Candidate summary (smoke)
```
head_name     head_role  mean_corrected_attention_spike  mean_intact_effect  mean_conditional_effect  mean_conditional_gain
     L9H0        backup                        0.000000           -0.108456                -0.098102               0.010354
     L9H7        backup                        0.000000            0.363768                 0.430905               0.067137
    L10H1        backup                        0.308376           -0.022008                 0.059219               0.081227
    L10H2        backup                        0.313392            0.111290                 0.320454               0.209164
    L10H6        backup                        0.273759            0.197388                 0.201248               0.003860
   L10H10        backup                        0.330846            0.212200                 0.122318              -0.089883
    L11H2        backup                        0.156342           -0.322580                -0.166623               0.155956
    L11H9        backup                        0.105520           -0.074855                -0.102866              -0.028010
    L10H7      negative                       -0.441598           -1.691377                -0.201883               1.489494
   L11H10      negative                        0.008847           -0.789710                -0.871827              -0.082118
    L10H3 extra_control                        0.186981            0.119643                 0.154225               0.034582
    L10H5 extra_control                       -0.001622           -0.036616                -0.029253               0.007363
```

- Runtime seconds: 13.62
- N=8 smoke only; do not interpret scientifically.
