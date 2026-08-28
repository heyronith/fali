# Phase-3C validation report

- device: cuda
- N=8 seed=42
- max z recon err: 4.768e-07
- max H000/intact err: 5.722e-06
- max H111/conditional err: 6.676e-06
- max gain identity err: 7.629e-06
- max Shapley residual: 4.441e-16

## Checks
- PASS: `no_nan_inf`
- PASS: `z_reconstruction_ok`
- PASS: `endpoint_identity_ok`
- PASS: `endpoint_preferred_tol`
- PASS: `shapley_residual_ok`
- PASS: `all_12_heads`
- PASS: `n_rows`
- PASS: `l9h7_pv_invariant`
- PASS: `historical_untouched`
- PASS: `cuda_ok`

## Factorial summary

head_name     head_role  mean_intact_effect  mean_conditional_effect  mean_conditional_gain  mean_phi_pattern  mean_phi_value  mean_phi_context  fraction_pattern_positive  fraction_value_positive  fraction_context_positive  max_endpoint_error  max_shapley_residual  existing_end_query_causal_total  mean_phi_pattern_minus_end_query
     L9H0        backup           -0.096337                -0.073029               0.023307          0.000000        0.000000          0.023309                      0.000                    0.000                      0.500            0.000006          1.387779e-17                         0.000000                          0.000000
     L9H7        backup            0.339345                 0.404538               0.065193          0.000000        0.000000          0.065194                      0.000                    0.000                      0.750            0.000005          0.000000e+00                         0.000000                          0.000000
    L10H1        backup           -0.031719                 0.004196               0.035915         -0.019842       -0.000131          0.055887                      0.375                    0.375                      0.750            0.000007          1.110223e-16                        -0.013635                         -0.006207
    L10H2        backup            0.148903                 0.499727               0.350824          0.318219        0.000896          0.031709                      1.000                    0.500                      0.375            0.000007          3.469447e-18                         0.331568                         -0.013349
    L10H6        backup            0.216416                 0.207995              -0.008421          0.109596        0.007823         -0.125840                      0.375                    1.000                      0.125            0.000008          5.551115e-17                         0.100927                          0.008669
   L10H10        backup            0.266508                 0.335844               0.069336          0.203100        0.009025         -0.142788                      0.625                    1.000                      0.125            0.000004          2.220446e-16                         0.169567                          0.033533
    L11H2        backup           -0.273879                -0.335454              -0.061575         -0.063286        0.008941         -0.007230                      0.375                    0.625                      0.375            0.000005          2.220446e-16                        -0.054708                         -0.008578
    L11H9        backup           -0.070076                -0.114473              -0.044397         -0.058407        0.001174          0.012837                      0.250                    0.750                      0.500            0.000004          6.938894e-18                        -0.054756                         -0.003651
    L10H7      negative           -1.289985                -0.284616               1.005369          1.193637        0.004423         -0.192691                      1.000                    0.500                      0.000            0.000005          4.440892e-16                         1.319574                         -0.125937
   L11H10      negative           -0.490327                -0.458133               0.032194          0.052605        0.013264         -0.033676                      0.625                    0.625                      0.250            0.000004          2.775558e-17                         0.053114                         -0.000508
    L10H3 extra_control            0.069736                 0.067827              -0.001909          0.045459        0.000536         -0.047903                      0.750                    0.750                      0.375            0.000006          2.775558e-17                         0.036886                          0.008572
    L10H5 extra_control           -0.033267                -0.040787              -0.007520          0.002995       -0.002572         -0.007944                      0.625                    0.125                      0.500            0.000006          6.938894e-18                         0.001945                          0.001050
