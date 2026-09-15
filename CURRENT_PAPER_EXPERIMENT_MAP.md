# CURRENT PAPER EXPERIMENT MAP

**Paper:** Factorized Attention-Level Intervention (FALI)  
**Manuscript:** `manuscript/neural_networks/`  
**Date:** 2026-09-15

This map proves which scripts/results support the current Neural Networks manuscript.

## Extended-FALI formalization / validation

| Phase | Manuscript claim | Script | Module | Config / plan | Seeds | Raw | Aggregate | Validation | Plot | Figure | MS location |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EF-0R | Standard Owen (36 perm); coarse–fine | `tests/test_extended_fali_math.py` | `src/extended_fali/` (Owen) | `docs/EXTENDED_FALI_PREREGISTRATION.md` | n/a | n/a | n/a | test suite | n/a | n/a | Method / Owen |
| EF-1 / EF-1R | Synthetic factor isolation; transformer invariants | `experiments/extended_fali_ef1_synthetic.py` | `src/extended_fali/` | prereg + formal spec | 4242 (TL) | `results/extended_fali/ef1_synthetic/` | same | `validation_report.json` | figure scripts under archive ML | Fig3 | Method Validation |
| EF-2 | GPT-2 smoke (pipeline) | `experiments/extended_fali_ef2_gpt2_smoke.py` / `modal_extended_fali_ef2.py` | extended_fali | run configs in results | smoke | `ef2_gpt2_smoke/` | smoke summaries | validation | n/a | n/a | provenance only |
| EF-3 | Two-template screen (pre-canonical) | `extended_fali_ef3_*.py` / modal ef3 | extended_fali | results configs | 42,1,8,22 | `ef3_two_template/` | aggregate | validation | n/a | mentioned in GPT-2 text | GPT-2 (historical screen) |

## GPT-2 IOI

| Phase | Claim | Script | Seeds | Results | MS |
|---|---|---|---|---|---|
| Phase-1 corrected conditional gain | L10H2 recruited; L10H10 not (broad) | `experiments/revision_phase1_*.py`, `modal_phase1.py` | 42,1,8,22 | `results/revision_phase1/` | GPT-2 functional |
| EF-4 canonical | $\phi_T>0$, $\phi_R<0$ L10H2/L10H10 | `extended_fali_ef4_canonical.py`, `ef4_analyze.py`, `modal_extended_fali_ef4.py` | 42,1,8,22 | `results/extended_fali/ef4_canonical/` | GPT-2 FALI; Fig4 |
| Architectural controls | L9H7/L9H0 → C only | EF-4 pipeline | same | EF-4 aggregate / validation | GPT-2 |
| EF-5 matched robustness | fine channels preserve; **overall gate FAIL** | `extended_fali_ef5_*.py`, `modal_extended_fali_ef5.py` | 42,1,8,22 | `ef5_ablation_robustness/`, `ef5_reference_diagnosis*` | Ablation section |
| Phase-4 figures (if used) | figure assets | `revision_phase4_*.py` | — | `results/revision_phase4/` | Fig provenance / supplement |

## Pythia induction

| Phase | Claim | Script | Seeds | Results | MS |
|---|---|---|---|---|---|
| EF-6 discovery | select L5H0 / L8H2 | `extended_fali_ef6_discovery.py` | 42 | `ef6_pythia_induction/discovery/` | Pythia design |
| EF-6 confirmation | recruit YES; $T<0$, $R>0$; transfer FAIL | `extended_fali_ef6_confirm.py`, `ef6_analyze.py`, `modal_extended_fali_ef6.py` | 1,8,22 × offsets | `ef6_pythia_induction/` | Pythia; Fig5 |

## Gauge sensitivity

| Item | Path |
|---|---|
| Preregistration | commit `e60f81a` / `preregistration/gauge_sensitivity_rowmean.md` (or docs) |
| Native primary | EF-4 / EF-6 aggregates (native logits) |
| Row-mean runs | `results/extended_fali/gauge_sensitivity/ef4_rowmean/`, `ef6_rowmean/` |
| Report | `GAUGE_SENSITIVITY_REPORT.md` (commit `4313519`) |
| Summary/paired | `gauge_sensitivity_summary.json`, `gauge_sensitivity_paired_differences.csv` |
| Manuscript | Ablation and Gauge Robustness; supplement tables |

## Failed evidence that MUST be retained

- EF-5 overall gate FAIL (`ef5_ablation_robustness/aggregate/`, reference diagnosis)
- EF-6 mechanism-transfer FAIL (`ef6_pythia_induction/aggregate/ef6_gate.json`)
- EF-0R supersession note (docs + tests)

## Not current-paper (Frozen Attention / superseded)

- `experiments/qk_decomposition.py`, `output_mediation.py`, `fali_decomposition.py` (attention-probability era)
- `results/raw/*` Frozen dumps
- `paper.tex`, `submission_tmlr/`, TMLR rejected archive (provenance only)
