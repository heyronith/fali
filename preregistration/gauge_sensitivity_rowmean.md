# Preregistered FALI Softmax-Gauge Sensitivity Analysis

**Status:** FROZEN BEFORE EXECUTION  
**Branch:** `experiment/gauge-sensitivity`  
**Base commit:** `9672c029ad64dcce746b103125ff5f4e6619f842`  
**Canonical gauge:** valid-key row-mean centering (`row_mean`)  
**Confirmatory transforms allowed:** this gauge only (no post-hoc alternatives)

---

## Scientific question

Do the published fine \(T/R/O\) Owen attribution conclusions survive when raw
attention-score rows are represented in a canonical zero-mean gauge?

Softmax obeys
\[
\operatorname{softmax}(s)=\operatorname{softmax}(s+c\cdot\mathbf{1}),
\]
so raw attention logits are identifiable only up to an additive row constant.

The **native** FALI analysis remains the preregistered primary analysis.
This sensitivity analysis asks whether the fine \(T/R/O\) decomposition is robust
to choosing a unique, permutation-symmetric representative of each
softmax-equivalence class.

---

## Canonical gauge

For every example and endpoint score row, over structurally valid keys only:

\[
s_{\mathrm{centered}}=s-\operatorname{mean}(s_{\mathrm{valid}}).
\]

Requirements:

- center endpoint \(A\) and endpoint \(B\) separately;
- mean uses valid keys only;
- masked/invalid positions do not enter the mean;
- masked/invalid positions remain masked;
- no target/rival-specific normalization;
- no fitting/tuning based on results.

This is the **only** confirmatory gauge transform in this run.

---

## Experiments

### GPT-2 EF-4 sensitivity

- GPT-2 Small; exact existing IOI configuration
- same perturbation, utility, \(T/R/O/V/C\) semantics
- candidates: **L10H2**, **L10H10**
- seeds: **42, 1, 8, 22**
- 900 examples/seed
- same 32 cells; Owen partition \(\{\{T,R,O\},\{V\},\{C\}\}\); 36 permutations
- same hierarchical bootstrap design as EF-4

### Pythia EF-6 sensitivity

- `EleutherAI/pythia-160m` with exact frozen revision from EF-6
- primary **L5H0**; candidate **L8H2**
- confirmation seeds only: **1, 8, 22**
- 384 examples/seed; offsets \(\{8,16,24\}\)
- same target-logprob utility, 32 cells, Owen, confirmation bootstrap
- **Do not rerun discovery.**

---

## Primary sensitivity outcomes

For each candidate report native and row-mean:

- \(\phi_T,\phi_R,\phi_O,\phi_V,\phi_C,\Phi_{\mathrm{SCORE}}\), gain

and paired differences

\[
\Delta\phi_X=\phi_X(\mathrm{rowmean})-\phi_X(\mathrm{native})
\]

for \(X\in\{T,R,O,V,C,\mathrm{SCORE}\}\).

---

## Prespecified interpretation

For each fine factor \(T\) and \(R\):

- **PRESERVED:** balanced mean has same sign as native; preregistered 95% CI excludes zero in same direction
- **SIGN-UNCERTAIN:** mean has same sign but CI includes zero
- **SIGN-CHANGED:** balanced mean changes sign

The central cross-setting fine-score contrast is **preserved** only if:

- GPT-2 L10H2: \(T>0\) and \(R<0\)
- GPT-2 L10H10: \(T>0\) and \(R<0\)
- Pythia L8H2: \(T<0\) and \(R>0\)

under the row-mean gauge with directional CIs excluding zero.

If this fails: report that the fine mechanism claim is **representation-sensitive**.
Do **not** reinterpret or search for another gauge.

Conditional recruitment/gain is treated separately and is expected to remain
unchanged because endpoint model behavior is gauge invariant.

---

## Claim boundary

Even if signs survive row-centering, do **not** claim universal gauge invariance.

Allowed:
> The fine-channel result is robust to the prespecified zero-mean gauge sensitivity.

Not allowed:
> The fine-channel decomposition is representation invariant.

---

## Threat model / gaming control

Primary risk: researcher degrees of freedom in choosing a gauge after observing the answer.

Mitigation:

- freeze row-mean gauge before execution;
- commit this preregistration before GPU run;
- record preregistration commit SHA in every run manifest;
- no post-hoc alternative gauges in the confirmatory analysis.

---

## Output locations

- `results/extended_fali/gauge_sensitivity/ef4_rowmean/`
- `results/extended_fali/gauge_sensitivity/ef6_rowmean/`

Native EF-4 / EF-6 trees must remain untouched.
