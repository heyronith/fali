# Phase-4A validation report

- device: cuda
- N=900 seed=8 (30×30 strata)
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
     L9H0        backup           -0.035573                 0.002837               0.038410          0.000000        0.000000          0.038410                   0.000000                 0.000000                   0.665556                 0.0          5.551115e-17                       0.0
     L9H7        backup            0.392005                 0.448044               0.056039          0.000000        0.000000          0.056039                   0.000000                 0.000000                   0.640000                 0.0          5.551115e-17                       0.0
    L10H1        backup            0.011830                -0.071742              -0.083572          0.057734        0.001004         -0.142310                   0.690000                 0.561111                   0.324444                 0.0          2.220446e-16                       0.0
    L10H2        backup            0.090717                 0.395213               0.304497          0.448115       -0.000603         -0.143015                   0.930000                 0.512222                   0.283333                 0.0          4.440892e-16                       0.0
    L10H6        backup            0.348388                 0.350035               0.001647          0.245797       -0.000334         -0.243816                   0.913333                 0.511111                   0.127778                 0.0          2.220446e-16                       0.0
   L10H10        backup            0.175074                -0.070205              -0.245279          0.211927        0.001986         -0.459192                   0.825556                 0.565556                   0.077778                 0.0          4.440892e-16                       0.0
    L11H2        backup           -0.505137                 0.158112               0.663249          0.667070       -0.006105          0.002284                   0.786667                 0.463333                   0.527778                 0.0          8.881784e-16                       0.0
    L11H9        backup           -0.089054                -0.027740               0.061314          0.066125        0.000642         -0.005453                   0.628889                 0.566667                   0.461111                 0.0          2.220446e-16                       0.0
    L10H7      negative           -1.851601                 0.284217               2.135818          1.943900        0.003668          0.188251                   0.988889                 0.648889                   0.694444                 0.0          1.776357e-15                       0.0
   L11H10      negative           -1.083934                -1.353087              -0.269153         -0.163592        0.001548         -0.107110                   0.401111                 0.538889                   0.084444                 0.0          4.440892e-16                       0.0
    L10H3 extra_control            0.146455                 0.089058              -0.057398          0.059886        0.001126         -0.118410                   0.836667                 0.592222                   0.237778                 0.0          1.110223e-16                       0.0
    L10H5 extra_control           -0.066388                -0.014102               0.052286          0.004652       -0.001222          0.048855                   0.637778                 0.385556                   0.777778                 0.0          5.551115e-17                       0.0
