# Phase-2 FALI target recommendation (contrastive panel)

Recommend a **contrastive panel**, not a single favorite head.
These recommendations are derived from corrected Phase-1C labels.

## Recommended panel

### L10H2
- Role in panel: Putative strong genuine recruited backup
- Numbers: intact=0.1170, cond_effect=0.3173, cond_gain=0.2003, spike=0.3047
- Falsifiable contrast: Tests whether FALI target-logit / QK changes accompany true positive conditional effect + gain.

### L10H1
- Role in panel: Possible sign-flip recruit (`suggestive_suppressive_to_supportive`; class=`suggestive_positive_functional_recruitment`)
- Numbers: intact=-0.0179, cond_effect=0.0523, cond_gain=0.0702, spike=0.3006
- Falsifiable contrast: Falsifies ‘spike = recruitment’ only if FALI shows routing while function flips or remains weak; or supports a clean recruit if both effect and gain are positive.

### L10H10
- Role in panel: Large attention spike but downregulated function (`supportive_but_downregulated`)
- Numbers: intact=0.2162, cond_effect=0.1300, cond_gain=-0.0862, spike=0.3226
- Falsifiable contrast: Critical negative control for attention-based self-repair claims: spike without recruitment.

### L9H7
- Role in panel: Functional change with zero same-layer attention spike (`strong_positive_functional_recruitment`)
- Numbers: intact=0.3512, cond_effect=0.4233, cond_gain=0.0720, spike=0.0000
- Falsifiable contrast: Tests whether FALI/QK can detect recruitment when attention spike is architecturally invisible.

### L10H7
- Role in panel: Suppressor / less-suppressive control (`less_suppressive_after_knockout`)
- Numbers: intact=-1.6787, cond_effect=-0.1997, cond_gain=1.4790, spike=-0.4351
- Falsifiable contrast: Must not be labeled a backup recruit; FALI should reflect suppressor dynamics / reduced suppression.

## Why not only L10H10

Under corrected primary knockout, L10H10 remains helpful but becomes *less* important (supportive but downregulated). Using it alone as the FALI exemplar would overfit the old narrative.

## Stop

Do not begin Phase 2 until research-lead review of this panel.

