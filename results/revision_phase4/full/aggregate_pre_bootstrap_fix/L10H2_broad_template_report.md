# L10H2 broad-template report

Framing: **primary H2 — pattern recruitment (+ secondary S2)**

## Hypothesis quantities

       hypothesis_id head_name           metric predicted_direction  primary  aggregate_mean  hierarchical_cluster_ci_low  hierarchical_cluster_ci_high  n_seeds_correct_direction  n_strata_correct_direction  n_strata_total               all30_status        nonoverlap29_status  nonoverlap29_mean  nonoverlap29_ci_low  nonoverlap29_ci_high
 H2_conditional_gain     L10H2 conditional_gain                   >     True        0.298451                     0.262867                      0.335261                          4                          26              30 STRONGLY_BROADLY_SUPPORTED STRONGLY_BROADLY_SUPPORTED           0.308133             0.259121              0.335821
      H2_phi_pattern     L10H2      phi_pattern                   >     True        0.445252                     0.409686                      0.477946                          4                          30              30 STRONGLY_BROADLY_SUPPORTED STRONGLY_BROADLY_SUPPORTED           0.449734             0.405742              0.477148
S2_L10H2_phi_context     L10H2      phi_context                   <    False       -0.146451                    -0.164744                     -0.128438                          4                          25              30 STRONGLY_BROADLY_SUPPORTED STRONGLY_BROADLY_SUPPORTED          -0.141158            -0.164937             -0.126316

## Hierarchical template-cluster aggregate (all 30 strata)

head_name head_role             metric  aggregate_mean    ci_low   ci_high  ci_excludes_zero_pos  ci_excludes_zero_neg  seed_mean_42  seed_mean_1  seed_mean_8  seed_mean_22  n_seeds_pos  n_seeds_neg                     procedure  n_bootstrap  primary_ci_for_generalization
    L10H2    backup conditional_effect        0.382239  0.342065  0.423694                  True                 False      0.396103     0.390696     0.395213      0.346943            4            0 hierarchical_template_cluster        10000                           True
    L10H2    backup   conditional_gain        0.298451  0.262867  0.335261                  True                 False      0.307963     0.304690     0.304497      0.276656            4            0 hierarchical_template_cluster        10000                           True
    L10H2    backup        phi_pattern        0.445252  0.409686  0.477946                  True                 False      0.451237     0.446303     0.448115      0.435351            4            0 hierarchical_template_cluster        10000                           True
    L10H2    backup          phi_value       -0.000349 -0.000802  0.000114                 False                 False     -0.000528    -0.000221    -0.000603     -0.000046            0            4 hierarchical_template_cluster        10000                           True
    L10H2    backup        phi_context       -0.146451 -0.164744 -0.128438                 False                  True     -0.142746    -0.141393    -0.143015     -0.158649            0            4 hierarchical_template_cluster        10000                           True
    L10H2    backup     interaction_PC       -0.039142 -0.050400 -0.028145                 False                  True     -0.040141    -0.035232    -0.040527     -0.040667            0            4 hierarchical_template_cluster        10000                           True

## Stratum-level means (n=120 per stratum)

template_stratum  conditional_effect  conditional_gain  phi_pattern  phi_value  phi_context  interaction_PC
        T00_ABBA            0.687431          0.613483     0.640676  -0.003768    -0.023425        0.026763
        T00_BABA            0.091317          0.017695     0.315259   0.002366    -0.299929       -0.110353
        T01_ABBA            0.604248          0.566098     0.606360  -0.002334    -0.037928       -0.011975
        T01_BABA            0.189212          0.007465     0.386889   0.004343    -0.383768       -0.109288
        T02_ABBA            0.376639          0.411062     0.465503  -0.002642    -0.051798       -0.016306
        T02_BABA            0.259181          0.081343     0.336247   0.001983    -0.256887       -0.079529
        T03_ABBA            0.336348          0.308387     0.311839  -0.003288    -0.000165       -0.005114
        T03_BABA            0.154959          0.031575     0.242189   0.002511    -0.213124       -0.076966
        T04_ABBA            0.765057          0.615162     0.722241  -0.003711    -0.103367       -0.047541
        T04_BABA           -0.082327         -0.036411     0.225524   0.003767    -0.265702       -0.099740
        T05_ABBA            0.673502          0.509911     0.505122  -0.003506     0.008295        0.052593
        T05_BABA            0.196317          0.078261     0.373972   0.001810    -0.297521       -0.086054
        T06_ABBA            0.746089          0.626584     0.627759  -0.005902     0.004727        0.017508
        T06_BABA            0.252903          0.145608     0.379474   0.002807    -0.236672       -0.078870
        T07_ABBA            0.611724          0.646749     0.629491  -0.004685     0.021944        0.000962
        T07_BABA            0.322563          0.100568     0.403523   0.001813    -0.304768       -0.081016
        T08_ABBA            0.442035          0.383200     0.407795  -0.002907    -0.021688        0.015668
        T08_BABA            0.249662          0.101847     0.338808   0.001504    -0.238464       -0.069047
        T09_ABBA            0.451569          0.379531     0.406184  -0.001923    -0.024730        0.035812
        T09_BABA            0.271485          0.160865     0.356282   0.001266    -0.196684       -0.067965
        T10_ABBA            0.996948          0.843766     0.829892  -0.004185     0.018059        0.072400
        T10_BABA           -0.011139         -0.007813     0.197428   0.002866    -0.208107       -0.077452
        T11_ABBA            0.931276          0.862796     0.839973  -0.003797     0.026620        0.072475
        T11_BABA           -0.008649         -0.071706     0.177698   0.004104    -0.253508       -0.110561
        T12_ABBA            0.750756          0.597881     0.707211  -0.004119    -0.105211       -0.045492
        T12_BABA           -0.048293         -0.044045     0.186223   0.004368    -0.234636       -0.090418
        T13_ABBA            0.539597          0.474067     0.560779  -0.001132    -0.085580        0.011985
        T13_BABA            0.062419          0.013736     0.301032   0.002223    -0.289520       -0.088167
        T14_ABBA            0.378506          0.342426     0.436944  -0.000426    -0.094091       -0.039787
        T14_BABA            0.275830          0.193451     0.439229   0.000111    -0.245889       -0.088776

## BABA vs ABBA aggregates

head_name head_role pattern_type    n  conditional_effect  conditional_gain  phi_pattern  phi_value  phi_context  interaction_PC  low_level_template_id metric  baba_mean  abba_mean  diff sign_reversal large_asymmetry
    L10H2    backup         BABA 1800            0.145029          0.051496     0.310652   0.002523    -0.261679       -0.087613                    NaN    NaN        NaN        NaN   NaN           NaN             NaN
    L10H2    backup         ABBA 1800            0.619448          0.545407     0.579851  -0.003222    -0.031223        0.009330                    NaN    NaN        NaN        NaN   NaN           NaN             NaN

_Do not interpret N=60 smoke; this report uses full N=900×4. Secondary hypotheses are confirmatory only._
