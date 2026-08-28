# Phase-1 smoke/validation report

## Dataset validation
- All examples passed S×2 / IO×1 / end==' to' checks.
- tokens device=cuda:0; model device=cuda:0

## Primary knockout
- mean A=2.874149, B_old=1.995775, B=2.748730
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
- A mean IO−S: 2.874149
- B_old mean IO−S: 1.995775
- B mean IO−S: 2.748730

## Candidate summary (smoke)
```
head_name     head_role  mean_corrected_attention_spike  mean_intact_effect  mean_conditional_effect  mean_conditional_gain
     L9H0        backup                        0.000000           -0.097704                -0.090826               0.006878
     L9H7        backup                        0.000000            0.340939                 0.416545               0.075606
    L10H1        backup                        0.300209           -0.014275                 0.039994               0.054268
    L10H2        backup                        0.300506            0.100991                 0.268725               0.167734
    L10H6        backup                        0.260756            0.306290                 0.334317               0.028026
   L10H10        backup                        0.322333            0.214760                 0.136737              -0.078024
    L11H2        backup                        0.158236           -0.336910                -0.212210               0.124700
    L11H9        backup                        0.101588           -0.075849                -0.108844              -0.032996
    L10H7      negative                       -0.445097           -1.726749                -0.197581               1.529167
   L11H10      negative                        0.014916           -0.780152                -0.893291              -0.113139
    L10H3 extra_control                        0.174045            0.113593                 0.143646               0.030053
    L10H5 extra_control                       -0.001597           -0.032039                -0.026070               0.005968
```

- Runtime seconds: 22.16
- N=8 smoke only; do not interpret scientifically.
