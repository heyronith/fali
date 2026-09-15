# MLJ FINAL EVIDENCE AUDIT

Scientific source: Neural Networks production HEAD `95358ad` + research-lead MLJ adaptations.

| Section | Value | Source | Verified |
|---|---|---|---|
| Method Validation | `0.455046` | ef1 validation_report.json | **YES** |
| Method Validation | `-0.085684` | ef1 | **YES** |
| Method Validation | `-0.540729` | ef1 | **YES** |
| GPT-2 | `0.382` | revision_phase4 / EF-4 | **YES** |
| GPT-2 | `0.298` | EF-4 | **YES** |
| GPT-2 | `0.783` | EF-4 phi_T | **YES** |
| GPT-2 | `-0.341` | EF-4 phi_R | **YES** |
| GPT-2 | `0.329` | EF-4 L10H10 | **YES** |
| Robustness | `0.440` | EF-5 matched mean | **YES** |
| Robustness | `-0.319` | EF-5 damage | **YES** |
| Gauge | `0.644` | gauge report | **YES** |
| Pythia | `14.017` | EF-6 | **YES** |
| Pythia | `-0.144` | EF-6 phi_T | **YES** |
| Pythia | `0.118` | EF-6 phi_R | **YES** |

**Verified quantitative claims:** 14/14

## Qualitative safeguards

| Claim | Verified |
|---|---|
| L10H2 positive recruitment; L10H10 not | **YES** |
| GPT-2 T>0, R<0 | **YES** |
| Pythia T<0, R>0; transfer failed | **YES** |
| Matched alternatives preserve fine signs; overall EF-5 gate failed | **YES** |
| Gauge: sign robustness only | **YES** |
| C coarse surrounding context | **YES** |
| No first-self-repair / first-conditional / new-Owen claims | **YES** |

## Unsupported claims found

None.
