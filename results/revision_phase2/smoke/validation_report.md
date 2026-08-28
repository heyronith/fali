# Phase 2A/3A validation report

- device: `cuda`
- N=8, seed=42, dtype=float32
- max |FALI residual|: 5.551e-17 (tol 1e-06)
- max |causal Shapley residual|: 0.000e+00 (tol 1e-05)
- max softmax-margin error: 9.516e-08 (tol 1e-05)
- L9H7 max |score_row A-B|: 0.000e+00
- L9H7 max |fali_target|: 0.000e+00
- L9H7 max |delta_attention|: 0.000e+00
- historical files unchanged: True

## Checks
- PASS: `no_nan_inf`
- PASS: `fali_residual_ok`
- PASS: `causal_residual_ok`
- PASS: `softmax_margin_ok`
- PASS: `all_12_descriptive`
- PASS: `all_6_causal`
- PASS: `l9h7_score_invariant`
- PASS: `l9h7_fali_near_zero`
- PASS: `l9h7_delta_attn_near_zero`
- PASS: `cuda_ok`
- PASS: `historical_untouched`

## Causal panel means (smoke — not for inference)
head_name head_role  layer  head      F00      F10      F01      F11  causal_target  causal_competing  causal_residual  pattern_clean_patch_effect  q_clean_effect  k_clean_effect  qk_clean_effect
    L10H2    backup     10     2 2.574738 3.155102 2.207608 2.906307       0.639531         -0.307962              0.0                    0.331569        0.319488        0.006237         0.330465
    L10H1    backup     10     1 2.919941 2.942638 2.861183 2.906307       0.033910         -0.047544              0.0                   -0.013634       -0.013582        0.000178        -0.013689
   L10H10    backup     10    10 2.736736 2.986688 2.580638 2.906307       0.287811         -0.118240              0.0                    0.169571        0.173343       -0.003514         0.169800
     L9H7    backup      9     7 2.906307 2.906307 2.906307 2.906307       0.000000          0.000000              0.0                    0.000000        0.000000        0.000000         0.000000
    L10H7  negative     10     7 1.586733 2.773749 1.651358 2.906307       1.220983          0.098591              0.0                    1.319574        1.322254        0.006254         1.320241
    L10H6    backup     10     6 2.805381 3.093697 2.534700 2.906307       0.329962         -0.229036              0.0                    0.100926        0.103518       -0.003523         0.100428

## Descriptive summary (selected)
head_name     head_role  delta_attention  delta_target_score  delta_competing_lse  delta_margin  fali_target  fali_competing  fali_residual
     L9H0        backup         0.000000            0.000000             0.000000      0.000000     0.000000        0.000000   0.000000e+00
     L9H7        backup         0.000000            0.000000             0.000000      0.000000     0.000000        0.000000   0.000000e+00
    L10H1        backup         0.258775            1.297904             0.110677      1.187226     0.280790       -0.022014   6.938894e-18
    L10H2        backup         0.305322            1.684474             0.291940      1.392535     0.364284       -0.058962   6.938894e-18
    L10H6        backup         0.240462            1.304095             0.196298      1.107797     0.280846       -0.040384   1.387779e-17
   L10H10        backup         0.301718            1.622662             0.248598      1.374064     0.352177       -0.050459   6.938894e-18
    L11H2        backup         0.123921            1.435495             0.066084      1.369410     0.130893       -0.006972   1.040834e-17
    L11H9        backup         0.083455            0.727464            -0.105455      0.832919     0.072501        0.010954   6.071532e-18
    L10H7      negative        -0.409597           -1.874613             0.054519     -1.929131    -0.397664       -0.011933   0.000000e+00
   L11H10      negative        -0.020122           -0.286321            -0.205217     -0.081104    -0.067297        0.047175   0.000000e+00
    L10H3 extra_control         0.164074            1.047390             0.282255      0.765134     0.221065       -0.056992  -1.734723e-18
    L10H5 extra_control        -0.002601           -0.610739             0.278562     -0.889301    -0.001835       -0.000766   2.981556e-19
