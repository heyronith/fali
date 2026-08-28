# L10H10 broad-template report (Phase-4C CI fix)

Framing: **primary H3 — cancellation (+ secondary S1)**

## Hypothesis quantities

           hypothesis_id head_name           metric predicted_direction  primary  aggregate_mean  hierarchical_cluster_ci_low  hierarchical_cluster_ci_high  n_seeds_correct_direction  n_strata_correct_direction  n_strata_total               all30_status        nonoverlap29_status  nonoverlap29_mean  nonoverlap29_ci_low  nonoverlap29_ci_high
          H3_phi_pattern    L10H10      phi_pattern                   >     True        0.206251                     0.185386                      0.225541                          4                          30              30 STRONGLY_BROADLY_SUPPORTED STRONGLY_BROADLY_SUPPORTED           0.205820             0.183168              0.226088
          H3_phi_context    L10H10      phi_context                   <     True       -0.471249                    -0.514346                     -0.426577                          4                          30              30 STRONGLY_BROADLY_SUPPORTED STRONGLY_BROADLY_SUPPORTED          -0.472160            -0.509458             -0.420323
     H3_conditional_gain    L10H10 conditional_gain                   <     True       -0.263442                    -0.301481                     -0.224900                          4                          30              30 STRONGLY_BROADLY_SUPPORTED STRONGLY_BROADLY_SUPPORTED          -0.264980            -0.297909             -0.218834
S1_L10H10_interaction_PC    L10H10   interaction_PC                   <    False       -0.334831                    -0.370729                     -0.297243                          4                          30              30 STRONGLY_BROADLY_SUPPORTED STRONGLY_BROADLY_SUPPORTED          -0.336517            -0.369942             -0.292686

## Hierarchical template-cluster aggregate (all 30 strata)

head_name head_role             metric  aggregate_mean    ci_low   ci_high  ci_excludes_zero_pos  ci_excludes_zero_neg  seed_mean_42  seed_mean_1  seed_mean_8  seed_mean_22  n_seeds_pos  n_seeds_neg                     procedure  n_bootstrap  primary_ci_for_generalization
   L10H10    backup conditional_effect       -0.078890 -0.111127 -0.045726                 False                  True     -0.067824    -0.072972    -0.070205     -0.104560            0            4 hierarchical_template_cluster        10000                           True
   L10H10    backup   conditional_gain       -0.263442 -0.301481 -0.224900                 False                  True     -0.258015    -0.259969    -0.245279     -0.290506            0            4 hierarchical_template_cluster        10000                           True
   L10H10    backup        phi_pattern        0.206251  0.185386  0.225541                  True                 False      0.212583     0.204122     0.211927      0.196372            4            0 hierarchical_template_cluster        10000                           True
   L10H10    backup          phi_value        0.001556  0.001012  0.002090                  True                 False      0.001258     0.001457     0.001986      0.001521            4            0 hierarchical_template_cluster        10000                           True
   L10H10    backup        phi_context       -0.471249 -0.514346 -0.426577                 False                  True     -0.471855    -0.465548    -0.459192     -0.488399            0            4 hierarchical_template_cluster        10000                           True
   L10H10    backup     interaction_PC       -0.334831 -0.370729 -0.297243                 False                  True     -0.333697    -0.326372    -0.332580     -0.346677            0            4 hierarchical_template_cluster        10000                           True

## Stratum-level means (n=120 per stratum)

template_stratum  conditional_effect  conditional_gain  phi_pattern  phi_value  phi_context  interaction_PC
        T00_ABBA           -0.258471         -0.457159     0.219593  -0.004140    -0.672612       -0.462350
        T00_BABA            0.004874         -0.218845     0.218748   0.007238    -0.444831       -0.285952
        T01_ABBA           -0.228778         -0.378548     0.206113  -0.003491    -0.581170       -0.404829
        T01_BABA           -0.017793         -0.379745     0.240686   0.005458    -0.625889       -0.343808
        T02_ABBA           -0.305573         -0.373970     0.140512  -0.002132    -0.512350       -0.302392
        T02_BABA           -0.004829         -0.233875     0.202932   0.004561    -0.441368       -0.267330
        T03_ABBA           -0.279647         -0.294378     0.066357  -0.001316    -0.359419       -0.180534
        T03_BABA            0.037745         -0.173215     0.159846   0.004305    -0.337366       -0.210081
        T04_ABBA           -0.316273         -0.511757     0.207284  -0.003464    -0.715577       -0.439336
        T04_BABA           -0.002882         -0.229945     0.184434   0.006768    -0.421147       -0.292449
        T05_ABBA           -0.237575         -0.310196     0.158720  -0.001378    -0.467538       -0.320573
        T05_BABA            0.059830         -0.213212     0.238967   0.006187    -0.458366       -0.275109
        T06_ABBA           -0.285707         -0.334792     0.179188  -0.003632    -0.510348       -0.430667
        T06_BABA            0.149017         -0.139949     0.239323   0.004272    -0.383544       -0.304323
        T07_ABBA           -0.132918         -0.313247     0.268789  -0.004308    -0.577728       -0.378617
        T07_BABA            0.184577         -0.258174     0.257679   0.004266    -0.520118       -0.272849
        T08_ABBA           -0.132737         -0.167475     0.138706  -0.001674    -0.304507       -0.216358
        T08_BABA            0.161600         -0.084428     0.225174   0.004805    -0.314406       -0.224115
        T09_ABBA           -0.128881         -0.212739     0.169420  -0.002480    -0.379679       -0.241754
        T09_BABA            0.044094         -0.142048     0.217018   0.005017    -0.364083       -0.260223
        T10_ABBA           -0.136467         -0.217885     0.289259  -0.004065    -0.503078       -0.488710
        T10_BABA            0.041607         -0.148695     0.158201   0.005233    -0.312129       -0.263865
        T11_ABBA           -0.210082         -0.318821     0.273310  -0.004476    -0.587655       -0.519983
        T11_BABA            0.148093         -0.093904     0.221623   0.009092    -0.324619       -0.267378
        T12_ABBA           -0.224733         -0.451591     0.207710  -0.003218    -0.656084       -0.444692
        T12_BABA           -0.042386         -0.189261     0.138626   0.009140    -0.337027       -0.238466
        T13_ABBA           -0.077977         -0.280790     0.283294  -0.001809    -0.562275       -0.521965
        T13_BABA           -0.105592         -0.295510     0.213490   0.005782    -0.514782       -0.405845
        T14_ABBA           -0.005107         -0.189131     0.203371  -0.001552    -0.390950       -0.346688
        T14_BABA           -0.063738         -0.289982     0.259154   0.007679    -0.556815       -0.433700

## BABA vs ABBA aggregates

head_name head_role pattern_type    n  conditional_effect  conditional_gain  phi_pattern  phi_value  phi_context  interaction_PC  low_level_template_id metric  baba_mean  abba_mean  diff sign_reversal large_asymmetry
   L10H10    backup         BABA 1800            0.039615         -0.206052     0.211727   0.005987    -0.423766       -0.289700                    NaN    NaN        NaN        NaN   NaN           NaN             NaN
   L10H10    backup         ABBA 1800           -0.197395         -0.320832     0.200775  -0.002876    -0.518731       -0.379963                    NaN    NaN        NaN        NaN   NaN           NaN             NaN

_Do not interpret N=60 smoke; this report uses full N=900×4. Secondary hypotheses are confirmatory only._
