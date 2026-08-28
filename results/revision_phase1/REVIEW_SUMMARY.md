# Phase-1A Review Summary

**Branch:** `revision/fali-validity`  
**Date:** 2026-08-13T19:06:16.765113+00:00  
**Stop condition:** N=8 smoke only. No N=200. No FALI Phase 2. No manuscript edits.

## 1. Files created / modified

### Created (new revision pipeline)
- `src/revision_config.py` — canonical head taxonomy
- `src/revision_interventions.py` — clone-safe composite zero-ablation hooks
- `src/revision_dataset.py` — validation + metadata wrapper around existing `IOIDataset`
- `src/revision_stats.py` — summaries + stratified bootstrap CI + correlations
- `experiments/revision_phase1_conditional_gain.py` — Phase-1 experiment CLI
- `tests/test_revision_phase1.py` — automated tests
- `results/revision_phase1/smoke/*` — smoke outputs only

### Modified for this task
- Device resolver in the new experiment defaults to **CPU** (avoids TransformerLens MPS correctness warning). No historical experiment scripts were edited for this revision task.

### Explicitly untouched
- Old experiment scripts under `experiments/` (except the new file)
- `src/ioi_dataset.py`
- `results/raw/`, `results/summary/`, `figures/`, `paper.tex`, manuscript text

## 2. Test results

```text
.venv/bin/python -m pytest tests/test_revision_phase1.py -v
13 passed
```

Coverage includes: composite hooks / no input mutation / one-hook-per-layer, metric identities, deterministic stratified bootstrap, dataset end/IO/S checks, output-path guards, smoke artifact presence.

## 3. Smoke command

```bash
.venv/bin/python experiments/revision_phase1_conditional_gain.py \
    --num-samples 8 \
    --seed 42 \
    --smoke \
    --device cpu \
    --output-dir results/revision_phase1/smoke
```

## 4. Smoke runtime environment

- Python: 3.10.20
- device: `cpu`
- torch: 2.12.0
- transformer-lens: 3.2.1
- transformers: 4.57.6
- platform: macOS-26.5.2-arm64-arm-64bit
- n_bootstrap (smoke): 200

## 5. Were any existing (pre-revision) source files modified?

No historical FALI experiment modules or `ioi_dataset.py` were modified for Phase-1A. Only new revision modules/tests/experiment were added (plus this review package).

## 6. Were any existing results overwritten?

**No.** Historical SHA256 hashes unchanged before/after smoke:

| File | Unchanged |
|---|---|
| `results/summary/attention_results.json` | True |
| `results/summary/ablation_controls_summary.json` | True |
| `results/summary/output_mediation_summary.json` | True |
| `results/raw/fali_decomposition_raw.csv` | True |

Overall `unchanged=True`

## 7. Observed smoke means (N=8 — not scientific)

| Condition | mean IO−S |
|---|---|
| A clean | 3.521108 |
| B_old (L9H9+L10H0) | 2.144467 |
| B corrected (L9H6+L9H9+L10H0) | 2.906314 |

## 8. Candidate smoke table (N=8 — not scientific)

```
head_name     head_role  mean_corrected_attention_spike  mean_intact_effect  mean_conditional_effect  mean_conditional_gain  conditional_gain_ci_low  conditional_gain_ci_high  fraction_positive_conditional_gain
     L9H0        backup                        0.000000           -0.096343                -0.073027               0.023316                -0.005197                  0.057495                               0.500
     L9H7        backup                        0.000000            0.339343                 0.404542               0.065199                 0.016121                  0.110896                               0.750
    L10H1        backup                        0.258776           -0.031717                 0.004205               0.035922                -0.122061                  0.167057                               0.500
    L10H2        backup                        0.305322            0.148902                 0.499732               0.350830                 0.144199                  0.519163                               0.875
    L10H6        backup                        0.240462            0.216411                 0.208003              -0.008408                -0.197568                  0.176314                               0.375
   L10H10        backup                        0.301719            0.266508                 0.335849               0.069342                -0.013246                  0.172810                               0.625
    L11H2        backup                        0.123921           -0.273879                -0.335449              -0.061570                -0.648844                  0.600804                               0.375
    L11H9        backup                        0.083455           -0.070080                -0.114467              -0.044387                -0.076805                 -0.015561                               0.250
    L10H7      negative                       -0.409597           -1.289984                -0.284610               1.005375                 0.733772                  1.299798                               0.875
   L11H10      negative                       -0.020122           -0.490329                -0.458128               0.032201                -0.055874                  0.121593                               0.625
    L10H3 extra_control                        0.164073            0.069735                 0.067835              -0.001901                -0.072056                  0.065546                               0.375
    L10H5 extra_control                       -0.002601           -0.033270                -0.040783              -0.007513                -0.029489                  0.011815                               0.500
```

## 9. Implementation concerns

1. **Do not interpret N=8.** Smoke only validates execution/algebra/I/O.
2. **Device default is CPU**, not MPS, because TransformerLens warns MPS may silently miscompute. Explicit `--device mps` still allowed.
3. On this smoke draw, **B_corrected > B_old** (ablating L9H6 *reduces* degradation vs historical knockout). L9H6-only ablation *increases* IO−S above clean in this tiny sample — consistent with L9H6 not behaving like a helpful primary on N=8, but **not** a claim.
4. Layer-9 candidate attention spikes under corrected primary knockout are **exactly 0** for L9H0/L9H7 in smoke — expected from same-layer parallel attention (z-ablation of other L9 heads does not change L9 QK).
5. Bootstrap CIs use **200** resamples in smoke (10,000 intended for full run).
6. `pytest` was installed into `.venv` to run tests (`python -m pip install pytest`).

## 10. ZIP path

`/Users/ronny/Desktop/Research/AI ALIGNMENT/FALI/phase1a_review_package.zip`
