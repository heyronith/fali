# L10H10 Phase-1 full-run report

## Per-seed

| seed | clean_attn | corrected_attn | spike | intact | conditional | gain | CI low | CI high | tmplA | tmplB |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.3779 | 0.6991 | 0.3212 | 0.2372 | 0.1631 | -0.0742 | -0.1196 | -0.0298 | -0.0694 | -0.0789 |
| 1 | 0.3733 | 0.6956 | 0.3223 | 0.2148 | 0.1367 | -0.0780 | -0.1234 | -0.0338 | -0.0041 | -0.1520 |
| 8 | 0.3671 | 0.6980 | 0.3308 | 0.2122 | 0.1223 | -0.0899 | -0.1345 | -0.0442 | -0.0425 | -0.1373 |
| 22 | 0.3724 | 0.6883 | 0.3159 | 0.2004 | 0.0979 | -0.1025 | -0.1475 | -0.0579 | -0.0766 | -0.1285 |

## Across-seed means

- mean attention spike: 0.322567
- mean intact effect: 0.216158
- mean conditional effect: 0.130005
- mean conditional gain: -0.086154
- seeds with positive gain: 0/4
- seeds with positive CI excluding zero: 0/4
- classification: `stable_negative_conditional_gain`

## Descriptive answer

> No: `L10H10` shows **stable negative** conditional gain under the corrected knockout.

This report does not discuss FALI.
