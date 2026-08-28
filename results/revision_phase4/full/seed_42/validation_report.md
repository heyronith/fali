# Phase-4A validation report

- device: cuda
- N=900 seed=42 (30×30 strata)
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
     L9H0        backup           -0.034387                 0.005461               0.039847          0.000000        0.000000          0.039847                   0.000000                 0.000000                   0.675556                 0.0          2.775558e-17                       0.0
     L9H7        backup            0.393051                 0.456071               0.063020          0.000000        0.000000          0.063020                   0.000000                 0.000000                   0.667778                 0.0          1.110223e-16                       0.0
    L10H1        backup            0.008475                -0.081894              -0.090369          0.056566        0.000874         -0.147809                   0.666667                 0.556667                   0.321111                 0.0          2.220446e-16                       0.0
    L10H2        backup            0.088140                 0.396103               0.307963          0.451237       -0.000528         -0.142746                   0.927778                 0.502222                   0.275556                 0.0          4.440892e-16                       0.0
    L10H6        backup            0.380260                 0.376956              -0.003304          0.252346       -0.001417         -0.254232                   0.917778                 0.458889                   0.130000                 0.0          2.220446e-16                       0.0
   L10H10        backup            0.190191                -0.067824              -0.258015          0.212583        0.001258         -0.471855                   0.816667                 0.536667                   0.065556                 0.0          4.440892e-16                       0.0
    L11H2        backup           -0.493319                 0.171459               0.664779          0.670507       -0.004105         -0.001623                   0.785556                 0.490000                   0.503333                 0.0          4.440892e-16                       0.0
    L11H9        backup           -0.083963                -0.013383               0.070581          0.076440        0.000732         -0.006592                   0.648889                 0.517778                   0.465556                 0.0          2.220446e-16                       0.0
    L10H7      negative           -1.907378                 0.285139               2.192517          1.987809        0.004230          0.200478                   0.987778                 0.644444                   0.710000                 0.0          8.881784e-16                       0.0
   L11H10      negative           -1.131739                -1.375516              -0.243777         -0.133065        0.001444         -0.112156                   0.413333                 0.510000                   0.061111                 0.0          4.440892e-16                       0.0
    L10H3 extra_control            0.151182                 0.102050              -0.049132          0.064022        0.000611         -0.113765                   0.823333                 0.563333                   0.194444                 0.0          1.110223e-16                       0.0
    L10H5 extra_control           -0.061041                -0.008713               0.052327          0.004002       -0.001011          0.049337                   0.622222                 0.397778                   0.784444                 0.0          5.551115e-17                       0.0
