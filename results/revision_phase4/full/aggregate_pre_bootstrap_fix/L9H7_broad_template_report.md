# L9H7 broad-template report

Framing: **primary H1 — context recruitment**

## Hypothesis quantities

 hypothesis_id head_name      metric predicted_direction  primary  aggregate_mean  hierarchical_cluster_ci_low  hierarchical_cluster_ci_high  n_seeds_correct_direction  n_strata_correct_direction  n_strata_total                 all30_status          nonoverlap29_status  nonoverlap29_mean  nonoverlap29_ci_low  nonoverlap29_ci_high
H1_phi_pattern      L9H7 phi_pattern         approx_zero     True        0.000000                     0.000000                      0.000000                          4                          30              30 ARCHITECTURAL_INVARIANT_PASS ARCHITECTURAL_INVARIANT_PASS           0.000000             0.000000               0.00000
  H1_phi_value      L9H7   phi_value         approx_zero     True        0.000000                     0.000000                      0.000000                          4                          30              30 ARCHITECTURAL_INVARIANT_PASS ARCHITECTURAL_INVARIANT_PASS           0.000000             0.000000               0.00000
H1_phi_context      L9H7 phi_context                   >     True        0.058479                     0.046539                      0.070186                          4                          26              30   STRONGLY_BROADLY_SUPPORTED   STRONGLY_BROADLY_SUPPORTED           0.056324             0.044845               0.07031

## Hierarchical template-cluster aggregate (all 30 strata)

head_name head_role             metric  aggregate_mean   ci_low  ci_high  ci_excludes_zero_pos  ci_excludes_zero_neg  seed_mean_42  seed_mean_1  seed_mean_8  seed_mean_22  n_seeds_pos  n_seeds_neg                     procedure  n_bootstrap  primary_ci_for_generalization
     L9H7    backup conditional_effect        0.451371 0.391790 0.519473                  True                 False      0.456071     0.439990     0.448044      0.461378            4            0 hierarchical_template_cluster        10000                           True
     L9H7    backup   conditional_gain        0.058479 0.046162 0.070192                  True                 False      0.063020     0.059498     0.056039      0.055358            4            0 hierarchical_template_cluster        10000                           True
     L9H7    backup        phi_pattern        0.000000 0.000000 0.000000                 False                 False      0.000000     0.000000     0.000000      0.000000            0            0 hierarchical_template_cluster        10000                           True
     L9H7    backup          phi_value        0.000000 0.000000 0.000000                 False                 False      0.000000     0.000000     0.000000      0.000000            0            0 hierarchical_template_cluster        10000                           True
     L9H7    backup        phi_context        0.058479 0.046539 0.070186                  True                 False      0.063020     0.059498     0.056039      0.055358            4            0 hierarchical_template_cluster        10000                           True
     L9H7    backup     interaction_PC        0.000000 0.000000 0.000000                 False                 False      0.000000     0.000000     0.000000      0.000000            0            0 hierarchical_template_cluster        10000                           True

## Stratum-level means (n=120 per stratum)

template_stratum  conditional_effect  conditional_gain  phi_pattern  phi_value  phi_context  interaction_PC
        T00_ABBA            0.344754          0.017595          0.0        0.0     0.017595             0.0
        T00_BABA            0.441245          0.120951          0.0        0.0     0.120951             0.0
        T01_ABBA            0.319818          0.001144          0.0        0.0     0.001144             0.0
        T01_BABA            0.526537          0.117411          0.0        0.0     0.117411             0.0
        T02_ABBA            0.493801          0.027699          0.0        0.0     0.027699             0.0
        T02_BABA            0.406682          0.108094          0.0        0.0     0.108094             0.0
        T03_ABBA            0.517099          0.010476          0.0        0.0     0.010476             0.0
        T03_BABA            0.341814          0.067304          0.0        0.0     0.067304             0.0
        T04_ABBA            0.117321          0.018638          0.0        0.0     0.018638             0.0
        T04_BABA            0.384748          0.054261          0.0        0.0     0.054261             0.0
        T05_ABBA            0.103263         -0.041698          0.0        0.0    -0.041698             0.0
        T05_BABA            0.535997          0.136389          0.0        0.0     0.136389             0.0
        T06_ABBA            0.596154          0.022458          0.0        0.0     0.022458             0.0
        T06_BABA            0.603320          0.107919          0.0        0.0     0.107919             0.0
        T07_ABBA            0.557571          0.030865          0.0        0.0     0.030865             0.0
        T07_BABA            0.731929          0.153075          0.0        0.0     0.153075             0.0
        T08_ABBA            0.224956         -0.015230          0.0        0.0    -0.015230             0.0
        T08_BABA            0.503307          0.047160          0.0        0.0     0.047160             0.0
        T09_ABBA            0.363728          0.022684          0.0        0.0     0.022684             0.0
        T09_BABA            0.570415          0.059180          0.0        0.0     0.059180             0.0
        T10_ABBA            0.300517          0.040862          0.0        0.0     0.040862             0.0
        T10_BABA            0.563316          0.118657          0.0        0.0     0.118657             0.0
        T11_ABBA            0.213213         -0.004493          0.0        0.0    -0.004493             0.0
        T11_BABA            0.599026          0.155482          0.0        0.0     0.155482             0.0
        T12_ABBA            0.164113          0.017284          0.0        0.0     0.017284             0.0
        T12_BABA            0.551800          0.128406          0.0        0.0     0.128406             0.0
        T13_ABBA            0.412105         -0.037564          0.0        0.0    -0.037564             0.0
        T13_BABA            0.570409          0.093950          0.0        0.0     0.093950             0.0
        T14_ABBA            0.791116          0.049850          0.0        0.0     0.049850             0.0
        T14_BABA            0.691049          0.125552          0.0        0.0     0.125552             0.0

## BABA vs ABBA aggregates

head_name head_role pattern_type    n  conditional_effect  conditional_gain  phi_pattern  phi_value  phi_context  interaction_PC  low_level_template_id metric  baba_mean  abba_mean  diff sign_reversal large_asymmetry
     L9H7    backup         BABA 1800            0.534773          0.106253          0.0        0.0     0.106253             0.0                    NaN    NaN        NaN        NaN   NaN           NaN             NaN
     L9H7    backup         ABBA 1800            0.367969          0.010705          0.0        0.0     0.010705             0.0                    NaN    NaN        NaN        NaN   NaN           NaN             NaN

_Do not interpret N=60 smoke; this report uses full N=900×4. Secondary hypotheses are confirmatory only._
