# Phase-1 smoke/validation report

## Dataset validation
- All examples passed S×2 / IO×1 / end==' to' checks.
- tokens device=cuda:0; model device=cuda:0

## Primary knockout
- mean A=2.956735, B_old=2.158286, B=2.813006
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
- A mean IO−S: 2.956735
- B_old mean IO−S: 2.158286
- B mean IO−S: 2.813006

## Candidate summary (smoke)
```
head_name     head_role  mean_corrected_attention_spike  mean_intact_effect  mean_conditional_effect  mean_conditional_gain
     L9H0        backup                        0.000000           -0.106153                -0.098182               0.007971
     L9H7        backup                        0.000000            0.365782                 0.445077               0.079295
    L10H1        backup                        0.292084           -0.043632                 0.020755               0.064387
    L10H2        backup                        0.305040            0.142272                 0.367585               0.225314
    L10H6        backup                        0.263668            0.233410                 0.231508              -0.001902
   L10H10        backup                        0.315894            0.200441                 0.097892              -0.102549
    L11H2        backup                        0.161719           -0.319102                -0.175755               0.143347
    L11H9        backup                        0.106991           -0.078169                -0.106319              -0.028151
    L10H7      negative                       -0.425345           -1.624943                -0.148514               1.476430
   L11H10      negative                       -0.000340           -0.795463                -0.856838              -0.061374
    L10H3 extra_control                        0.178084            0.109696                 0.148869               0.039173
    L10H5 extra_control                       -0.001640           -0.033470                -0.022867               0.010602
```

- Runtime seconds: 13.91
- N=8 smoke only; do not interpret scientifically.
