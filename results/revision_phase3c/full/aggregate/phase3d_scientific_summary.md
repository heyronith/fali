# Phase-3D scientific summary

**Predefined verdict: PARTIAL**

## Notes
- H1 L9H7 context recruitment: True
- H2 L10H2 pattern-mediated +gain: True
- H3 L10H10 cancellation: True
- H4 L10H7 adaptive-but-suppressive: True (cond_effect<0=True)
- Template A/B major-head sign reversals: 1
- contrasts passed: 4/4; template_ok=False

## Value factor magnitude
{
  "mean_abs_phi_pattern": 0.20487741377618582,
  "mean_abs_phi_value": 0.003505083902014615,
  "mean_abs_phi_context": 0.07544692809383073,
  "value_over_pattern": 0.01710820064257387,
  "value_over_context": 0.04645760921711038
}

## Backup mechanism sketch (descriptive, not forced labels)
- L9H0: context
- L9H7: context
- L10H1: mixed
- L10H2: mixed
- L10H6: mixed
- L10H10: mixed
- L11H2: mixed
- L11H9: mixed

## Phase-2 end-query vs full-sequence pattern_effect_B_VB
               level seed  pearson_r    pearson_p  spearman_r  spearman_p
         across_head   42   0.999991 4.883130e-25         1.0         0.0
         across_head    1   0.999990 6.599971e-25         1.0         0.0
         across_head    8   0.999983 9.920057e-24         1.0         0.0
         across_head   22   0.999987 3.343873e-24         1.0         0.0
across_head_seedmean  all   0.999989 1.104006e-24         1.0         0.0

## Aggregate decomposition (selected)
head_name  mean_conditional_gain  mean_phi_pattern  mean_phi_value  mean_phi_context  mean_pattern_effect_B_VB
     L9H0               0.007137          0.000000        0.000000          0.007137                  0.000000
     L9H7               0.072038          0.000000        0.000000          0.072038                  0.000000
    L10H1               0.070233          0.021974       -0.001220          0.049479                  0.016740
    L10H2               0.200339          0.323728        0.001350         -0.124740                  0.305069
    L10H6               0.005906          0.202264        0.003727         -0.200084                  0.163735
   L10H10              -0.086154          0.222209        0.006612         -0.314975                  0.113663
    L11H2               0.114720          0.107448        0.016223         -0.008951                  0.125147
    L11H9              -0.030178         -0.032126        0.000101          0.001848                 -0.031124
    L10H7               1.479033          1.461977        0.002458          0.014598                  1.461522
   L11H10              -0.084036         -0.019576        0.007472         -0.071931                 -0.021514
    L10H3               0.037740          0.066079        0.000872         -0.029212                  0.056108
    L10H5               0.009492          0.001147       -0.002025          0.010370                 -0.002299

## Rules
- STRONG REPLACEMENT: H1–H4 reproduce with template-stable signs
- PARTIAL: some contrasts only
- FAILED PIVOT: smoke effects disappear / unstable
