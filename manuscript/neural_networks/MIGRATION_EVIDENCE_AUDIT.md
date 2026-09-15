# MIGRATION EVIDENCE AUDIT

**Date:** 2026-09-15  
**Scientific source:** `manuscript/archive/machine_learning/`  
**Destination:** `manuscript/neural_networks/`  
**Policy:** no unsupported numerical statement may enter the manuscript.

## Empirical claims (major)

| Section | Claim | Exact result source | Numerical value(s) | Verified |
|---|---|---|---|---|
| Method validation | Single-factor synthetic recovers manipulated factor | archive `method_validation.tex` / EF-1R synthetic | e.g. $\Delta_h=0.268952$, $\phi_T=0.268952$; max err $1.11\times10^{-16}$ | YES |
| Method validation | Random transformer endpoints | archive `method_validation.tex` | $E_A=0.455046$, $E_B=-0.085684$, $\Delta=-0.540729$ | YES |
| GPT-2 | L10H2 positive recruitment | EF-4 / archive `gpt2_ioi.tex` | $E_B=0.382$ $[0.341,0.424]$; gain $0.298$ $[0.263,0.335]$ | YES |
| GPT-2 | L10H10 attention↑ without recruitment | EF-4 / archive | $\Delta$attn $0.310$; gain $-0.263$ | YES |
| GPT-2 | L10H2 $T>0$, $R<0$ | EF-4 / archive | $\phi_T=0.783$ $[0.722,0.838]$; $\phi_R=-0.341$ $[-0.372,-0.311]$ | YES |
| GPT-2 | L10H10 $T>0$, $R<0$ | EF-4 / archive | $\phi_T=0.329$; $\phi_R=-0.126$ | YES |
| Ablation | Matched-mean/resample preserve fine signs | EF-5 / archive | e.g. L10H2 matched-mean $\phi_T=0.440$, $\phi_R=-0.166$ | YES |
| Ablation | Overall EF-5 gate FAIL (zero-ref) | EF-5 / archive | ref $-0.319$ $[-0.396,-0.244]$ | YES |
| Pythia | Positive recruitment | EF-6 / archive | gain $0.486$ $[0.326,0.722]$ | YES |
| Pythia | Fine signs reverse $T<0$, $R>0$ | EF-6 / archive | $\phi_T=-0.144$; $\phi_R=+0.118$ | YES |
| Gauge (pending commit 3) | Signs preserved under row-mean | commit `4313519` `GAUGE_SENSITIVITY_REPORT.md` | GPT-2 $T>0,R<0$; Pythia $T<0,R>0$ | PENDING |

## Novelty / literature bounds

| Claim | Citation(s) | Appropriately bounded? |
|---|---|---|
| Not first self-repair / conditional importance | Hydra, CoAx, etc. | YES — explicit non-claims retained |
| Narrow “to our knowledge” FALI estimand claim | Gong, Vaidyanathan, Wang, Zhu | YES — “to our knowledge” preserved |
| Not new Owen mathematics | Owen 1977 | YES |
| Not automatic circuit discovery | Limitations | YES |
| $C$ = surrounding-network/context residual | Related Work / Limitations | YES — not “downstream sensitivity” |

## Unsupported numerical statements found during migration

None introduced. All migrated numbers match the Factorized archive prose.

## Notes

- Historical Frozen Attention Logit Intervention sources were not used as scientific content.
- Gauge-sensitivity numbers are integrated only after verification against commit `4313519` (separate commit).
