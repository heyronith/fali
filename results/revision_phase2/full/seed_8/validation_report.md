# Phase 2/3 validation report

- device: `cuda`
- N=200, seed=8, dtype=float32
- max |FALI residual|: 1.110e-16 (tol 1e-06)
- max |causal Shapley residual|: 0.000e+00 (tol 1e-05)
- max softmax-margin error: 1.718e-07 (tol 1e-05)
- L9H7 max |score_row A-B|: 0.000e+00
- L9H7 max |fali_target|: 0.000e+00
- L9H7 max |delta_attention|: 0.000e+00
- historical files unchanged: True
- reporting panel (subset): ['L10H2', 'L10H1', 'L10H10', 'L9H7', 'L10H7', 'L10H6']

## Checks
- PASS: `no_nan_inf`
- PASS: `fali_residual_ok`
- PASS: `causal_residual_ok`
- PASS: `softmax_margin_ok`
- PASS: `all_12_descriptive`
- PASS: `all_12_causal`
- PASS: `n_rows_match`
- PASS: `l9h7_score_invariant`
- PASS: `l9h7_fali_near_zero`
- PASS: `l9h7_delta_attn_near_zero`
- PASS: `cuda_ok`
- PASS: `historical_untouched`

## Causal means (all 12 heads)
 seed head_name     head_role  layer  head      F00      F10      F01      F11  causal_target  causal_competing  causal_total  causal_residual  pattern_clean_patch_effect  q_clean_effect  k_clean_effect  qk_clean_effect
    8      L9H0        backup      9     0 2.808078 2.808078 2.808078 2.808078       0.000000          0.000000      0.000000              0.0                    0.000000        0.000000        0.000000         0.000000
    8      L9H7        backup      9     7 2.808078 2.808078 2.808078 2.808078       0.000000          0.000000      0.000000              0.0                    0.000000        0.000000        0.000000         0.000000
    8     L10H1        backup     10     1 2.785703 2.852723 2.714914 2.808078       0.080092         -0.057717      0.022376              0.0                    0.022376        0.022434       -0.000294         0.022388
    8     L10H2        backup     10     2 2.489997 3.083011 2.128513 2.808078       0.636290         -0.318208      0.318082              0.0                    0.318082        0.316310       -0.004054         0.317010
    8     L10H6        backup     10     6 2.648053 3.012579 2.352399 2.808078       0.410102         -0.250077      0.160025              0.0                    0.160025        0.161232       -0.003646         0.159559
    8    L10H10        backup     10    10 2.688739 2.908827 2.543276 2.808078       0.242445         -0.123106      0.119339              0.0                    0.119339        0.120820       -0.003520         0.119370
    8     L11H2        backup     11     2 2.642104 3.486175 1.902741 2.808078       0.874704         -0.708730      0.165974              0.0                    0.165974        0.170990       -0.016802         0.165974
    8     L11H9        backup     11     9 2.838781 2.857573 2.781968 2.808078       0.022451         -0.053154     -0.030703              0.0                   -0.030703       -0.030215       -0.000950        -0.030703
    8     L10H7      negative     10     7 1.303391 2.602437 1.327947 2.808078       1.389589          0.115099      1.504687              0.0                    1.504687        1.505373        0.007568         1.505506
    8    L11H10      negative     11    10 2.827974 2.872137 2.770802 2.808078       0.040720         -0.060615     -0.019895              0.0                   -0.019895       -0.032331        0.018056        -0.019895
    8     L10H3 extra_control     10     3 2.751810 2.845971 2.716268 2.808078       0.092986         -0.036718      0.056268              0.0                    0.056268        0.055417        0.000351         0.056712
    8     L10H5 extra_control     10     5 2.809806 2.809687 2.808170 2.808078      -0.000106         -0.001622     -0.001728              0.0                   -0.001728       -0.001244       -0.000614        -0.001896

## Descriptive summary
head_name     head_role  delta_attention  delta_target_score  delta_competing_lse  delta_margin  fali_target  fali_competing  fali_residual
     L9H0        backup         0.000000            0.000000             0.000000      0.000000     0.000000        0.000000   0.000000e+00
     L9H7        backup         0.000000            0.000000             0.000000      0.000000     0.000000        0.000000   0.000000e+00
    L10H1        backup         0.308376            1.573061             0.160744      1.412317     0.338918       -0.030543   2.983724e-18
    L10H2        backup         0.313392            1.780276             0.367396      1.412881     0.385575       -0.072183   2.914335e-18
    L10H6        backup         0.273759            1.575660             0.216894      1.358766     0.314415       -0.040657  -2.706169e-18
   L10H10        backup         0.330846            1.797112             0.317190      1.479922     0.393492       -0.062646  -4.857226e-19
    L11H2        backup         0.156342            1.508612             0.104383      1.404229     0.168470       -0.012128  -6.245005e-19
    L11H9        backup         0.105520            0.807208            -0.099833      0.907042     0.093277        0.012243  -1.040834e-19
    L10H7      negative        -0.441598           -2.139735             0.037289     -2.177024    -0.436046       -0.005551   7.632783e-19
   L11H10      negative         0.008847           -0.065215            -0.110649      0.045434    -0.014568        0.023416   0.000000e+00
    L10H3 extra_control         0.186981            1.219847             0.328377      0.891470     0.249256       -0.062275  -6.591949e-19
    L10H5 extra_control        -0.001622           -0.616857             0.231047     -0.847904    -0.001188       -0.000434   1.246832e-20
