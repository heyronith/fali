# Phase 2/3 validation report

- device: `cuda`
- N=200, seed=1, dtype=float32
- max |FALI residual|: 1.110e-16 (tol 1e-06)
- max |causal Shapley residual|: 0.000e+00 (tol 1e-05)
- max softmax-margin error: 1.634e-07 (tol 1e-05)
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
 seed head_name     head_role  layer  head      F00      F10      F01     F11  causal_target  causal_competing  causal_total  causal_residual  pattern_clean_patch_effect  q_clean_effect  k_clean_effect  qk_clean_effect
    1      L9H0        backup      9     0 2.748730 2.748730 2.748730 2.74873       0.000000          0.000000      0.000000              0.0                    0.000000        0.000000        0.000000         0.000000
    1      L9H7        backup      9     7 2.748730 2.748730 2.748730 2.74873       0.000000          0.000000      0.000000              0.0                    0.000000        0.000000        0.000000         0.000000
    1     L10H1        backup     10     1 2.737405 2.783720 2.679136 2.74873       0.057954         -0.046629      0.011325              0.0                    0.011325        0.011341       -0.000204         0.011353
    1     L10H2        backup     10     2 2.466832 3.046514 2.076466 2.74873       0.625973         -0.344075      0.281899              0.0                    0.281899        0.281434       -0.004896         0.280889
    1     L10H6        backup     10     6 2.563807 2.942935 2.278457 2.74873       0.424700         -0.239777      0.184923              0.0                    0.184923        0.187608       -0.004857         0.184042
    1    L10H10        backup     10    10 2.630781 2.840783 2.506641 2.74873       0.226045         -0.108096      0.117949              0.0                    0.117949        0.119550       -0.003659         0.117915
    1     L11H2        backup     11     2 2.611428 3.440980 1.852772 2.74873       0.862755         -0.725453      0.137303              0.0                    0.137303        0.141157       -0.014881         0.137303
    1     L11H9        backup     11     9 2.781892 2.797127 2.727310 2.74873       0.018328         -0.051490     -0.033162              0.0                   -0.033162       -0.032737       -0.000893        -0.033162
    1     L10H7      negative     10     7 1.208043 2.522079 1.260205 2.74873       1.401280          0.139407      1.540687              0.0                    1.540687        1.541281        0.007468         1.541471
    1    L11H10      negative     11    10 2.799607 2.837672 2.717354 2.74873       0.034720         -0.085597     -0.050877              0.0                   -0.050877       -0.062812        0.017351        -0.050877
    1     L10H3 extra_control     10     3 2.698748 2.788919 2.662119 2.74873       0.088391         -0.038409      0.049982              0.0                    0.049982        0.049117        0.000282         0.050338
    1     L10H5 extra_control     10     5 2.751827 2.751671 2.748852 2.74873      -0.000139         -0.002958     -0.003097              0.0                   -0.003097       -0.002584       -0.000604        -0.003243

## Descriptive summary
head_name     head_role  delta_attention  delta_target_score  delta_competing_lse  delta_margin  fali_target  fali_competing  fali_residual
     L9H0        backup         0.000000            0.000000             0.000000      0.000000     0.000000        0.000000   0.000000e+00
     L9H7        backup         0.000000            0.000000             0.000000      0.000000     0.000000        0.000000   0.000000e+00
    L10H1        backup         0.300209            1.517148             0.143787      1.373361     0.327664       -0.027455  -4.163336e-19
    L10H2        backup         0.300506            1.756943             0.391859      1.365084     0.377615       -0.077109  -2.220446e-18
    L10H6        backup         0.260756            1.508260             0.228729      1.279531     0.303992       -0.043236  -6.938894e-20
   L10H10        backup         0.322333            1.744537             0.310126      1.434411     0.384658       -0.062325   9.714451e-19
    L11H2        backup         0.158236            1.537200             0.110570      1.426629     0.170673       -0.012438  -1.387779e-18
    L11H9        backup         0.101588            0.805811            -0.096103      0.901914     0.090317        0.011271  -3.122502e-19
    L10H7      negative        -0.445097           -2.125993             0.068086     -2.194079    -0.432347       -0.012750   2.359224e-18
   L11H10      negative         0.014916           -0.054925            -0.123420      0.068495    -0.012438        0.027354  -6.938894e-20
    L10H3 extra_control         0.174045            1.176051             0.326113      0.849938     0.234175       -0.060130  -2.428613e-19
    L10H5 extra_control        -0.001597           -0.600602             0.228718     -0.829320    -0.001163       -0.000434   5.963112e-21
