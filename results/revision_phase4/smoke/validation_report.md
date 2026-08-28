# Phase-4A validation report

- device: cuda
- N=60 seed=42 (2×30 strata)
- max z recon err: 0.000e+00
- max H000/intact err: 0.000e+00
- max H111/conditional err: 0.000e+00
- max gain identity err: 0.000e+00
- max Shapley residual: 8.882e-16
- max interaction recon residual: 0.000e+00

## Checks
- PASS: `no_nan_inf`
- PASS: `z_reconstruction_ok`
- PASS: `endpoint_identity_ok`
- PASS: `endpoint_preferred_tol`
- PASS: `shapley_residual_ok`
- PASS: `interaction_identity_ok`
- PASS: `all_12_heads`
- PASS: `n_rows`
- PASS: `all_30_strata`
- PASS: `l9_pv_invariant`
- PASS: `historical_untouched`
- PASS: `cuda_ok`

## Factorial summary (smoke — not for scientific inference)

head_name     head_role  mean_intact_effect  mean_conditional_effect  mean_conditional_gain  mean_phi_pattern  mean_phi_value  mean_phi_context  fraction_pattern_positive  fraction_value_positive  fraction_context_positive  max_endpoint_error  max_shapley_residual  max_interaction_residual
     L9H0        backup           -0.046075                 0.001635               0.047710          0.000000        0.000000          0.047710                   0.000000                 0.000000                   0.650000                 0.0          2.775558e-17                       0.0
     L9H7        backup            0.339972                 0.365905               0.025933          0.000000        0.000000          0.025933                   0.000000                 0.000000                   0.566667                 0.0          2.775558e-17                       0.0
    L10H1        backup            0.037698                -0.007531              -0.045229          0.064033        0.000048         -0.109311                   0.750000                 0.466667                   0.400000                 0.0          1.110223e-16                       0.0
    L10H2        backup            0.049598                 0.372885               0.323287          0.470486       -0.002385         -0.144813                   0.900000                 0.483333                   0.233333                 0.0          2.220446e-16                       0.0
    L10H6        backup            0.372563                 0.361185              -0.011378          0.249316       -0.002769         -0.257925                   0.933333                 0.433333                   0.100000                 0.0          1.110223e-16                       0.0
   L10H10        backup            0.218361                -0.098927              -0.317288          0.214318        0.001151         -0.532757                   0.800000                 0.550000                   0.083333                 0.0          2.220446e-16                       0.0
    L11H2        backup           -0.608710                 0.015672               0.624381          0.636723       -0.000161         -0.012181                   0.816667                 0.483333                   0.433333                 0.0          2.220446e-16                       0.0
    L11H9        backup           -0.118520                -0.053245               0.065275          0.070645        0.001974         -0.007344                   0.633333                 0.566667                   0.416667                 0.0          5.551115e-17                       0.0
    L10H7      negative           -1.972486                 0.226159               2.198645          2.000647        0.002663          0.195335                   1.000000                 0.583333                   0.633333                 0.0          8.881784e-16                       0.0
   L11H10      negative           -1.076753                -1.363880              -0.287128         -0.189317        0.008252         -0.106063                   0.450000                 0.700000                   0.083333                 0.0          4.440892e-16                       0.0
    L10H3 extra_control            0.156270                 0.108648              -0.047622          0.059825        0.000808         -0.108256                   0.850000                 0.616667                   0.183333                 0.0          5.551115e-17                       0.0
    L10H5 extra_control           -0.061827                -0.005623               0.056204          0.006539       -0.000676          0.050341                   0.633333                 0.366667                   0.750000                 0.0          5.551115e-17                       0.0
