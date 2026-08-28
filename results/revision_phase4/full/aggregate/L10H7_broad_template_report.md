# L10H7 broad-template report (Phase-4C CI fix)

Framing: **primary H4 — adaptive-but-suppressive**

## Hypothesis quantities

        hypothesis_id head_name             metric predicted_direction  primary  aggregate_mean  hierarchical_cluster_ci_low  hierarchical_cluster_ci_high  n_seeds_correct_direction  n_strata_correct_direction  n_strata_total               all30_status        nonoverlap29_status  nonoverlap29_mean  nonoverlap29_ci_low  nonoverlap29_ci_high
       H4_phi_pattern     L10H7        phi_pattern                   >     True        1.965672                     1.845867                      2.058402                          4                          30              30 STRONGLY_BROADLY_SUPPORTED STRONGLY_BROADLY_SUPPORTED           1.973619             1.838296              2.058606
H4_conditional_effect     L10H7 conditional_effect                   <     True        0.303955                     0.252930                      0.353675                          0                           9              30      NOT_BROADLY_SUPPORTED      NOT_BROADLY_SUPPORTED           0.328085             0.256293              0.360737

## Hierarchical template-cluster aggregate (all 30 strata)

head_name head_role             metric  aggregate_mean   ci_low  ci_high  ci_excludes_zero_pos  ci_excludes_zero_neg  seed_mean_42  seed_mean_1  seed_mean_8  seed_mean_22  n_seeds_pos  n_seeds_neg                     procedure  n_bootstrap  primary_ci_for_generalization
    L10H7  negative conditional_effect        0.303955 0.252930 0.353675                  True                 False      0.285139     0.340493     0.284217      0.305970            4            0 hierarchical_template_cluster        10000                           True
    L10H7  negative   conditional_gain        2.166027 2.048833 2.267098                  True                 False      2.192517     2.186857     2.135818      2.148917            4            0 hierarchical_template_cluster        10000                           True
    L10H7  negative        phi_pattern        1.965672 1.845867 2.058402                  True                 False      1.987809     1.987943     1.943900      1.943038            4            0 hierarchical_template_cluster        10000                           True
    L10H7  negative          phi_value        0.004017 0.003300 0.004755                  True                 False      0.004230     0.004003     0.003668      0.004166            4            0 hierarchical_template_cluster        10000                           True
    L10H7  negative        phi_context        0.196338 0.172220 0.220787                  True                 False      0.200478     0.194910     0.188251      0.201714            4            0 hierarchical_template_cluster        10000                           True
    L10H7  negative     interaction_PC        0.041859 0.017276 0.068348                  True                 False      0.023511     0.059304     0.037422      0.047201            4            0 hierarchical_template_cluster        10000                           True

## Stratum-level means (n=120 per stratum)

template_stratum  conditional_effect  conditional_gain  phi_pattern  phi_value  phi_context  interaction_PC
        T00_ABBA            0.828600          2.662819     2.350234   0.007703     0.304883        0.215685
        T00_BABA           -0.395806          1.746424     1.735206  -0.000305     0.011523       -0.049013
        T01_ABBA            0.704801          2.587739     2.235512   0.006643     0.345584        0.035365
        T01_BABA           -0.251901          2.059492     1.963290   0.000668     0.095535        0.004048
        T02_ABBA            0.746595          2.148150     1.795772   0.005523     0.346855       -0.014846
        T02_BABA           -0.083579          1.797259     1.762517   0.001304     0.033438        0.019590
        T03_ABBA            0.686366          1.528806     1.176495   0.004511     0.347800       -0.010560
        T03_BABA           -0.106599          1.609830     1.480149   0.001244     0.128437        0.087040
        T04_ABBA            0.762042          2.911077     2.474312   0.009292     0.427473        0.026294
        T04_BABA           -0.433526          1.591741     1.625553  -0.000821    -0.032991       -0.003660
        T05_ABBA            0.942876          2.632339     2.313246   0.006486     0.312607        0.021975
        T05_BABA           -0.072726          2.017925     1.915697   0.002587     0.099640       -0.000421
        T06_ABBA            0.799220          2.629694     2.218765   0.007952     0.402977       -0.068084
        T06_BABA            0.052092          1.936594     1.807205  -0.000277     0.129665        0.066472
        T07_ABBA            0.685636          2.494271     2.195025   0.007422     0.291824       -0.022728
        T07_BABA            0.008202          2.004047     1.915199   0.002536     0.086313        0.110185
        T08_ABBA            0.680115          1.975420     1.787728   0.004424     0.183267        0.051536
        T08_BABA            0.138349          1.990067     1.788081   0.002487     0.199499       -0.025785
        T09_ABBA            0.614796          2.090175     1.958361   0.006101     0.125713        0.078448
        T09_BABA            0.141674          2.045390     1.897502   0.002016     0.145871       -0.027668
        T10_ABBA            0.549491          2.598335     2.377720   0.008178     0.212438        0.217001
        T10_BABA           -0.326304          1.440000     1.377799   0.003160     0.059042       -0.026086
        T11_ABBA            0.897100          2.787663     2.533767   0.007163     0.246734        0.288306
        T11_BABA           -0.303513          1.727416     1.720641   0.000491     0.006283       -0.026089
        T12_ABBA            0.896440          3.087152     2.687742   0.008627     0.390782        0.204461
        T12_BABA           -0.315390          1.542792     1.483213  -0.000229     0.059809       -0.066492
        T13_ABBA            0.529520          2.412802     2.144281   0.002131     0.266390        0.195277
        T13_BABA            0.185661          2.070982     1.955503  -0.000142     0.115621        0.040999
        T14_ABBA            0.554842          2.545596     2.152812   0.007839     0.384945       -0.068776
        T14_BABA            0.003571          2.308827     2.140843   0.005792     0.162193        0.003305

## BABA vs ABBA aggregates

head_name head_role pattern_type    n  conditional_effect  conditional_gain  phi_pattern  phi_value  phi_context  interaction_PC  low_level_template_id metric  baba_mean  abba_mean  diff sign_reversal large_asymmetry
    L10H7  negative         BABA 1800           -0.117320          1.859252     1.771226   0.001367     0.086658        0.007095                    NaN    NaN        NaN        NaN   NaN           NaN             NaN
    L10H7  negative         ABBA 1800            0.725229          2.472803     2.160118   0.006666     0.306018        0.076624                    NaN    NaN        NaN        NaN   NaN           NaN             NaN

_Do not interpret N=60 smoke; this report uses full N=900×4. Secondary hypotheses are confirmatory only._

## Taxonomy vs measured role

Historical Phase-1 taxonomy may label this head a Negative Name Mover; measured conditional_effect on the canonical-template / public-implementation lexical distribution is positive on aggregate. Both facts are reported without contradiction.
