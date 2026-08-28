# PHASE4C Review Summary — Hierarchical Bootstrap Fix + Role Audit

**Date:** 2026-08-14  
**Model execution:** none (raw Phase-4B factorial reused)  
**Verdict (unchanged rule):** **BROAD-TEMPLATE PARTIAL**

## Bootstrap fix

`hierarchical_template_cluster_bootstrap_means` now draws

`ex_idx[bootstrap, seed, template_slot, pattern, example]`

so within-cell prompt resampling is **independent per seed**, while template clusters remain **shared** across seeds within a draw. Seeds are not resampled; four seed means remain equally weighted.

Pre-fix artifacts preserved under `results/revision_phase4/full/aggregate_pre_bootstrap_fix/`.

## Tests

`test_hierarchical_bootstrap_independent_per_seed_resampling` — **PASS**  
(proves shared clusters, BABA/ABBA pairing, per-seed independent `ex_idx`, SHA256 determinism, equal weighting; buggy shared-`ex_idx` fails independence).

## Old vs fixed CIs

- Means unchanged (max |Δ| ~ 1e-16)
- **No hypothesis status changes**
- Largest |CI width change|: **0.0031** (`H4_phi_pattern`)
- Corrected verdict: **BROAD-TEMPLATE PARTIAL**

## Functional-role audit (tol=0.05)

Historical taxonomy kept separate from measured role.

Notable BABA/ABBA role changes among audited heads: L9H7, L10H10, L10H7, L10H1, L11H2, L10H6.  
L10H2 stable `supportive_recruited` on both patterns.

L10H7: historical Negative Name Mover; measured aggregate conditional_effect **positive**; BABA net-suppressive-less-suppressive vs ABBA supportive_recruited → template-dependent measured role.

## Package

`phase4c_review_package.zip`
