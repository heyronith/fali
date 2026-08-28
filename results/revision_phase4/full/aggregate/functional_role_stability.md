# Functional role stability (Phase-4C audit)

Near-zero / mixed tolerance: |conditional_effect| or |conditional_gain| < 0.05
→ classified `near_zero_or_mixed` (not treated as a meaningful role).

Historical taxonomy labels (e.g. `negative`) are provenance from Phase-1,
not a claim that the head is negative on every prompt distribution.

## 1. Heads with stable functional role across ≥24/30 strata

- L10H1: no single meaningful role ≥24/30 (top=`suppressive_more_suppressive` 15/30)
- L10H10: no single meaningful role ≥24/30 (top=`suppressive_more_suppressive` 16/30)
- L10H2: no single meaningful role ≥24/30 (top=`supportive_recruited` 22/30)
- L10H3: no single meaningful role ≥24/30 (top=`near_zero_or_mixed` 21/30)
- L10H5: no single meaningful role ≥24/30 (top=`near_zero_or_mixed` 26/30)
- L10H6: no single meaningful role ≥24/30 (top=`near_zero_or_mixed` 15/30)
- L10H7: no single meaningful role ≥24/30 (top=`supportive_recruited` 19/30)
- L11H10: no single meaningful role ≥24/30 (top=`suppressive_more_suppressive` 23/30)
- L11H2: no single meaningful role ≥24/30 (top=`supportive_recruited` 18/30)
- L11H9: no single meaningful role ≥24/30 (top=`supportive_recruited` 14/30)
- L9H0: no single meaningful role ≥24/30 (top=`near_zero_or_mixed` 24/30)
- L9H7: no single meaningful role ≥24/30 (top=`near_zero_or_mixed` 16/30)

## 2. Strong BABA/ABBA role reversals

- **L9H7**: BABA=`supportive_recruited` vs ABBA=`near_zero_or_mixed` (hist taxonomy=`backup`)
- L10H2: same role `supportive_recruited` on BABA and ABBA (hist taxonomy=`backup`)
- **L10H10**: BABA=`near_zero_or_mixed` vs ABBA=`suppressive_more_suppressive` (hist taxonomy=`backup`)
- **L10H7**: BABA=`suppressive_less_suppressive` vs ABBA=`supportive_recruited` (hist taxonomy=`negative`)
- **L10H1**: BABA=`supportive_recruited` vs ABBA=`suppressive_more_suppressive` (hist taxonomy=`backup`)
- **L11H2**: BABA=`suppressive_less_suppressive` vs ABBA=`supportive_recruited` (hist taxonomy=`backup`)
- **L10H6**: BABA=`supportive_downregulated` vs ABBA=`supportive_recruited` (hist taxonomy=`backup`)

## 3. Old two-template role failed to generalize?

- L10H7 H4 conditional_effect<0: **NOT_BROADLY_SUPPORTED** (mean=0.3040). Historically a Negative Name Mover; measured conditional effect on canonical IOI is positive.
- L10H1 suggestive φ_context>0: **NOT_BROADLY_SUPPORTED** (mean=-0.1462).

## 4. Are L9H7 and L10H2 still genuine recruits under broad templates?

- H1_phi_context: STRONGLY_BROADLY_SUPPORTED (mean=0.0585)
- H2_conditional_gain: STRONGLY_BROADLY_SUPPORTED (mean=0.2985)
- H2_phi_pattern: STRONGLY_BROADLY_SUPPORTED (mean=0.4453)
- L9H7 strata classified supportive_recruited: 14/30
- L10H2 strata classified supportive_recruited: 22/30

## 5. L10H10 canonical role description

functional_role
suppressive_more_suppressive    16
near_zero_or_mixed               9
supportive_downregulated         5

Measured pattern: φ_P>0, φ_C<0, gain<0 (H3 strongly supported). This is better described as **positive pattern adaptation overwhelmed by context** rather than merely 'helpful backup downregulated', because the head's pattern contribution remains positive while context dominates the net conditional gain.

## 6. Does L10H7 demonstrate template-dependent functional role?

functional_role
supportive_recruited            19
suppressive_less_suppressive     9
near_zero_or_mixed               2

- BABA role=`suppressive_less_suppressive` (cond=-0.117, gain=1.859)
- ABBA role=`supportive_recruited` (cond=0.725, gain=2.473)
- Historical taxonomy=`negative` (Negative Name Mover provenance); measured canonical conditional_effect is positive on aggregate — report both without contradiction.

