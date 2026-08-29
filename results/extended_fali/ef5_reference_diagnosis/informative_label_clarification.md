# EF-5 informative-label clarification (post-stop, does not alter EF-5 artifacts)

The preregistered EF-5 gate remains **FAIL** with `reference_perturbation_stop = true`.

Zero reference balanced mean damage_drop = **-0.318825** (not positive with CI > 0).

Therefore criterion C (`alternative_mean >= 0.25 * zero_mean`) is **NOT_APPLICABLE** when `zero_mean < 0` — the fractional reference is not scientifically interpretable.

Post-stop descriptive labels for alternatives:

- **matched_mean**: `performance_degrading_AB = true`; `fractional_reference_C = NOT_APPLICABLE`
- **matched_resample**: `performance_degrading_AB = true`; `fractional_reference_C = NOT_APPLICABLE`

Original EF-5 `INFORMATIVE` fields in aggregate artifacts are unchanged.
