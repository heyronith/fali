# L10H1 broad-template report (Phase-4C CI fix)

Framing: **secondary / suggestive**

## Hypothesis quantities

              hypothesis_id head_name      metric predicted_direction  primary  aggregate_mean  hierarchical_cluster_ci_low  hierarchical_cluster_ci_high  n_seeds_correct_direction  n_strata_correct_direction  n_strata_total          all30_status   nonoverlap29_status  nonoverlap29_mean  nonoverlap29_ci_low  nonoverlap29_ci_high
secondary_L10H1_phi_context     L10H1 phi_context                   >    False        -0.14617                    -0.160483                     -0.131401                          0                           7              30 NOT_BROADLY_SUPPORTED NOT_BROADLY_SUPPORTED          -0.154902            -0.162675             -0.132594

## Hierarchical template-cluster aggregate (all 30 strata)

head_name head_role             metric  aggregate_mean    ci_low   ci_high  ci_excludes_zero_pos  ci_excludes_zero_neg  seed_mean_42  seed_mean_1  seed_mean_8  seed_mean_22  n_seeds_pos  n_seeds_neg                     procedure  n_bootstrap  primary_ci_for_generalization
    L10H1    backup conditional_effect       -0.076414 -0.102423 -0.051201                 False                  True     -0.081894    -0.083873    -0.071742     -0.068147            0            4 hierarchical_template_cluster        10000                           True
    L10H1    backup   conditional_gain       -0.087487 -0.103707 -0.070561                 False                  True     -0.090369    -0.092080    -0.083572     -0.083928            0            4 hierarchical_template_cluster        10000                           True
    L10H1    backup        phi_pattern        0.057843  0.049113  0.066046                  True                 False      0.056566     0.057095     0.057734      0.059978            4            0 hierarchical_template_cluster        10000                           True
    L10H1    backup          phi_value        0.000839  0.000429  0.001317                  True                 False      0.000874     0.000669     0.001004      0.000811            4            0 hierarchical_template_cluster        10000                           True
    L10H1    backup        phi_context       -0.146170 -0.160483 -0.131401                 False                  True     -0.147809    -0.149843    -0.142310     -0.144716            0            4 hierarchical_template_cluster        10000                           True
    L10H1    backup     interaction_PC       -0.063958 -0.071916 -0.056289                 False                  True     -0.063850    -0.064903    -0.063544     -0.063537            0            4 hierarchical_template_cluster        10000                           True

## Stratum-level means (n=120 per stratum)

template_stratum  conditional_effect  conditional_gain  phi_pattern  phi_value  phi_context  interaction_PC
        T00_ABBA           -0.379427         -0.325484     0.020449   0.002964    -0.348897       -0.122166
        T00_BABA            0.330850          0.228914     0.124410  -0.002572     0.107077       -0.003307
        T01_ABBA           -0.337091         -0.311692     0.024054   0.003842    -0.339588       -0.100820
        T01_BABA            0.244735          0.165014     0.119248  -0.003561     0.049327       -0.012586
        T02_ABBA           -0.241934         -0.250224     0.039487   0.003067    -0.292778       -0.073034
        T02_BABA            0.220272          0.121154     0.088722  -0.001558     0.033990       -0.019726
        T03_ABBA           -0.321410         -0.253007     0.007575   0.003486    -0.264068       -0.057400
        T03_BABA            0.133131          0.034598     0.057490  -0.001457    -0.021435       -0.027477
        T04_ABBA           -0.330579         -0.298366     0.027304   0.003701    -0.329371       -0.097290
        T04_BABA            0.293283          0.207288     0.081900  -0.002254     0.127642        0.002433
        T05_ABBA           -0.399946         -0.347363    -0.000257   0.003484    -0.350590       -0.113898
        T05_BABA            0.172295          0.112911     0.123009  -0.003120    -0.006977       -0.034732
        T06_ABBA           -0.267429         -0.255198     0.043628   0.002252    -0.301078       -0.133066
        T06_BABA            0.152684          0.096030     0.112322  -0.000314    -0.015978       -0.030009
        T07_ABBA           -0.307686         -0.303304     0.029157   0.002133    -0.334594       -0.118657
        T07_BABA            0.068087          0.049503     0.079510   0.000131    -0.030139       -0.038465
        T08_ABBA           -0.234063         -0.224979     0.028995   0.000567    -0.254540       -0.074100
        T08_BABA            0.076638          0.024860     0.081877  -0.000812    -0.056204       -0.035548
        T09_ABBA           -0.214926         -0.233161     0.023750   0.001866    -0.258778       -0.075140
        T09_BABA            0.127779          0.054888     0.100602  -0.000637    -0.045077       -0.025585
        T10_ABBA           -0.302444         -0.243425     0.026287   0.001238    -0.270950       -0.116777
        T10_BABA            0.158583          0.114543     0.067398  -0.000785     0.047929       -0.010579
        T11_ABBA           -0.322929         -0.270946     0.015058   0.003706    -0.289710       -0.132971
        T11_BABA            0.198232          0.124430     0.081575  -0.002386     0.045241       -0.007687
        T12_ABBA           -0.557273         -0.455675    -0.018588   0.003602    -0.440688       -0.132362
        T12_BABA            0.262421          0.196655     0.069763  -0.001474     0.128367       -0.006001
        T13_ABBA           -0.329234         -0.272487     0.017774   0.003307    -0.293568       -0.139133
        T13_BABA            0.095691          0.099136     0.129021   0.000486    -0.030370       -0.051344
        T14_ABBA           -0.282038         -0.217268     0.017557   0.005296    -0.240121       -0.070361
        T14_BABA            0.001303          0.008040     0.116223   0.000980    -0.109163       -0.060961

## BABA vs ABBA aggregates

head_name head_role pattern_type    n  conditional_effect  conditional_gain  phi_pattern  phi_value  phi_context  interaction_PC  low_level_template_id metric  baba_mean  abba_mean  diff sign_reversal large_asymmetry
    L10H1    backup         BABA 1800            0.169066          0.109198     0.095538  -0.001289     0.014949       -0.024105                    NaN    NaN        NaN        NaN   NaN           NaN             NaN
    L10H1    backup         ABBA 1800           -0.321894         -0.284172     0.020149   0.002967    -0.307288       -0.103812                    NaN    NaN        NaN        NaN   NaN           NaN             NaN

_Do not interpret N=60 smoke; this report uses full N=900×4. Secondary hypotheses are confirmatory only._
