# Phase-4A validation report

- device: cuda
- N=900 seed=22 (30×30 strata)
- max z recon err: 0.000e+00
- max H000/intact err: 0.000e+00
- max H111/conditional err: 0.000e+00
- max gain identity err: 0.000e+00
- max Shapley residual: 1.776e-15
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
     L9H0        backup           -0.042420                -0.008900               0.033520          0.000000        0.000000          0.033520                   0.000000                 0.000000                   0.628889                 0.0          5.551115e-17                       0.0
     L9H7        backup            0.406021                 0.461378               0.055358          0.000000        0.000000          0.055358                   0.000000                 0.000000                   0.624444                 0.0          5.551115e-17                       0.0
    L10H1        backup            0.015780                -0.068147              -0.083928          0.059978        0.000811         -0.144716                   0.687778                 0.538889                   0.301111                 0.0          2.220446e-16                       0.0
    L10H2        backup            0.070287                 0.346943               0.276656          0.435351       -0.000046         -0.158649                   0.937778                 0.508889                   0.254444                 0.0          4.440892e-16                       0.0
    L10H6        backup            0.346467                 0.339068              -0.007399          0.242792       -0.000629         -0.249562                   0.917778                 0.480000                   0.146667                 0.0          2.220446e-16                       0.0
   L10H10        backup            0.185946                -0.104560              -0.290506          0.196372        0.001521         -0.488399                   0.821111                 0.542222                   0.056667                 0.0          4.440892e-16                       0.0
    L11H2        backup           -0.483523                 0.244252               0.727775          0.727524       -0.008695          0.008946                   0.805556                 0.458889                   0.548889                 0.0          8.881784e-16                       0.0
    L11H9        backup           -0.099518                -0.032745               0.066773          0.074830        0.000332         -0.008389                   0.661111                 0.521111                   0.447778                 0.0          2.220446e-16                       0.0
    L10H7      negative           -1.842948                 0.305970               2.148917          1.943038        0.004166          0.201714                   0.993333                 0.650000                   0.742222                 0.0          1.776357e-15                       0.0
   L11H10      negative           -1.084279                -1.358424              -0.274145         -0.163509        0.001666         -0.112303                   0.401111                 0.501111                   0.068889                 0.0          4.440892e-16                       0.0
    L10H3 extra_control            0.154346                 0.094121              -0.060225          0.061786        0.001079         -0.123090                   0.846667                 0.601111                   0.193333                 0.0          1.110223e-16                       0.0
    L10H5 extra_control           -0.064058                -0.012756               0.051302          0.003748       -0.000970          0.048524                   0.622222                 0.406667                   0.788889                 0.0          5.551115e-17                       0.0
