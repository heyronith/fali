# Phase-4A validation report

- device: cuda
- N=900 seed=1 (30×30 strata)
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
     L9H0        backup           -0.037423                -0.001238               0.036185          0.000000        0.000000          0.036185                   0.000000                 0.000000                   0.658889                 0.0          5.551115e-17                       0.0
     L9H7        backup            0.380492                 0.439990               0.059498          0.000000        0.000000          0.059498                   0.000000                 0.000000                   0.635556                 0.0          1.110223e-16                       0.0
    L10H1        backup            0.008207                -0.083873              -0.092080          0.057095        0.000669         -0.149843                   0.684444                 0.547778                   0.332222                 0.0          2.220446e-16                       0.0
    L10H2        backup            0.086007                 0.390696               0.304690          0.446303       -0.000221         -0.141393                   0.936667                 0.508889                   0.288889                 0.0          4.440892e-16                       0.0
    L10H6        backup            0.378532                 0.376921              -0.001611          0.241307       -0.001093         -0.241825                   0.908889                 0.461111                   0.153333                 0.0          2.220446e-16                       0.0
   L10H10        backup            0.186997                -0.072972              -0.259969          0.204122        0.001457         -0.465548                   0.810000                 0.541111                   0.061111                 0.0          4.440892e-16                       0.0
    L11H2        backup           -0.503753                 0.187294               0.691047          0.695668       -0.005626          0.001006                   0.796667                 0.444444                   0.534444                 0.0          8.881784e-16                       0.0
    L11H9        backup           -0.087193                -0.016489               0.070703          0.079524        0.000594         -0.009415                   0.645556                 0.527778                   0.443333                 0.0          2.220446e-16                       0.0
    L10H7      negative           -1.846364                 0.340493               2.186857          1.987943        0.004003          0.194910                   0.988889                 0.645556                   0.715556                 0.0          1.776357e-15                       0.0
   L11H10      negative           -1.075324                -1.359065              -0.283740         -0.176333        0.001887         -0.109294                   0.391111                 0.515556                   0.078889                 0.0          4.440892e-16                       0.0
    L10H3 extra_control            0.148344                 0.097512              -0.050832          0.062233        0.000637         -0.113702                   0.820000                 0.547778                   0.192222                 0.0          1.110223e-16                       0.0
    L10H5 extra_control           -0.064154                -0.011195               0.052959          0.004460       -0.001098          0.049596                   0.610000                 0.385556                   0.778889                 0.0          5.551115e-17                       0.0
