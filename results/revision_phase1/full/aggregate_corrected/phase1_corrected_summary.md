# Phase-1 corrected summary

Interpretation rule: **positive conditional gain alone is not functional recruitment**.
Recruitment requires positive **conditional effect** as well.

## 1. Canonical backups that genuinely become more helpful

- **L10H2** (strong_positive_functional_recruitment): cond_effect=0.3173, cond_gain=0.2003, spike=0.3047
- **L9H7** (strong_positive_functional_recruitment): cond_effect=0.4233, cond_gain=0.0720, spike=0.0000
- **L10H1** (suggestive_positive_functional_recruitment): cond_effect=0.0523, cond_gain=0.0702, spike=0.3006

## 2. Remain helpful but become less important (downregulated)

- **L10H10**: cond_effect=0.1300, cond_gain=-0.0862, spike=0.3226

## 3. Remain suppressive despite positive gain (less-suppressive, not recruits)

- **L9H0** (backup): cond_effect=-0.0964, cond_gain=0.0071
- **L11H2** (backup): cond_effect=-0.2204, cond_gain=0.1147
- **L10H7** (negative): cond_effect=-0.1997, cond_gain=1.4790
- **L10H5** (extra_control): cond_effect=-0.0241, cond_gain=0.0095

## 4. Functional sign flips

- **L10H1**: `suggestive_suppressive_to_supportive` (intact=-0.0179, cond_effect=0.0523)

## 5. Does attention spike predict genuine recruitment?

- Among canonical backups, attention spike vs conditional gain: Pearson=0.07571225024988501, Spearman=-0.09581010095306072.
- Among canonical backups, attention spike vs conditional effect: Pearson=0.18200247209616133, Spearman=0.17964393928698885.
- Conclusion: a large attention spike is **not sufficient** for recruitment (see L10H10: large spike, positive conditional effect, but negative gain / downregulated). Absence of a spike is also not decisive for same-layer heads (see L9H7).

## Focused head reassessment

| head | role | intact | cond_effect | cond_gain | spike | sign_flip | class |
|---|---|---:|---:|---:|---:|---|---|
| L10H2 | backup | 0.1170 | 0.3173 | 0.2003 | 0.3047 | none | strong_positive_functional_recruitment |
| L10H1 | backup | -0.0179 | 0.0523 | 0.0702 | 0.3006 | suggestive_suppressive_to_supportive | suggestive_positive_functional_recruitment |
| L9H7 | backup | 0.3512 | 0.4233 | 0.0720 | 0.0000 | none | strong_positive_functional_recruitment |
| L10H10 | backup | 0.2162 | 0.1300 | -0.0862 | 0.3226 | none | supportive_but_downregulated |
| L11H2 | backup | -0.3351 | -0.2204 | 0.1147 | 0.1565 | none | less_suppressive_after_knockout |
| L10H7 | negative | -1.6787 | -0.1997 | 1.4790 | -0.4351 | none | less_suppressive_after_knockout |

## Full classification table

```
head_name     head_role  mean_intact_effect  mean_conditional_effect  mean_conditional_gain  mean_attention_spike                       sign_flip_type                 corrected_functional_class
     L9H0        backup           -0.103523                -0.096386               0.007137              0.000000                                 none            less_suppressive_after_knockout
     L9H7        backup            0.351228                 0.423266               0.072038              0.000000                                 none     strong_positive_functional_recruitment
    L10H1        backup           -0.017896                 0.052337               0.070233              0.300561 suggestive_suppressive_to_supportive suggestive_positive_functional_recruitment
    L10H2        backup            0.116970                 0.317309               0.200339              0.304732                                 none     strong_positive_functional_recruitment
    L10H6        backup            0.250665                 0.256571               0.005906              0.265062                                 none            supportive_no_stable_modulation
   L10H10        backup            0.216158                 0.130005              -0.086154              0.322567                                 none               supportive_but_downregulated
    L11H2        backup           -0.335096                -0.220376               0.114720              0.156501                                 none            less_suppressive_after_knockout
    L11H9        backup           -0.074699                -0.104877              -0.030178              0.103846                                 none            more_suppressive_after_knockout
    L10H7      negative           -1.678739                -0.199706               1.479033             -0.435124                                 none            less_suppressive_after_knockout
   L11H10      negative           -0.787778                -0.871813              -0.084036              0.008243                                 none            more_suppressive_after_knockout
    L10H3 extra_control            0.115111                 0.152850               0.037740              0.177502                                 none     strong_positive_functional_recruitment
    L10H5 extra_control           -0.033579                -0.024087               0.009492             -0.001649                                 none            less_suppressive_after_knockout
```

