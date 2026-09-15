# PROJECT CLEANUP REPORT

**Date:** 2026-09-15  
**Branch:** `maintenance/current-fali-cleanup`  
**Starting HEAD:** `bf6f89e470474a5be4def6dfa76ceada549224b0`  
**Safety tag:** `pre-current-fali-cleanup` → `bf6f89e470474a5be4def6dfa76ceada549224b0`

## Sizes

| Scope | Before | After |
|---|---|---|
| Full tree (`du -sh .`) | ~1.9 GB | ~1.7 GB |
| Except `.git` + `.venv` | 568,545,076 bytes (~542 MB) | 422,686,573 bytes (~403 MB) |

| Metric | Value |
|---|---|
| Bytes deleted (ops log) | 204,669,736 (~195 MB) |
| Bytes archived (ops log; includes copies) | 58,159,858 (~55 MB) |
| Files/dirs deleted (ops log entries) | 97 |
| Archive operations | 23 |
| Space reclaimed (excl. `.venv`, approx) | ~147 MB (542→402 MB) |
| Full-tree reclaimed | ~200 MB (1.9→1.7 GB; `.venv` ~1.2 GB retained) |

Ops detail: `PROJECT_CLEANUP_OPS.json`

## Major deleted categories

- Review/audit ZIPs and build/staging trees
- `submission_tmlr/` (+ tectonic caches) and `tmlr-style-file-main/`
- Local tooling: `tools/texlive-local/`, `tools/tectonic.tar.gz`, `tools/pydeps/`, `tools/springer-template/`
- Root `figures/`, `Hydra.pdf`, `IOI.pdf`, duplicate manuscript ZIPs
- Python/LaTeX caches and temps
- Phase review summary markdown (registry retained)

## Major retained historical files

| Path | Why |
|---|---|
| `archive/historical_fali/paper_frozen_attention.tex` | Supersession provenance cited in Related Work |
| `archive/historical_fali/tmlr_submission_main.tex` | One TMLR archival copy |
| `archive/historical_fali/experiments/*` | Frozen-era scripts (moved, not deleted) |
| `archive/manuscripts/tmlr_rejected_*` | Existing minimal TMLR archive |
| `results/raw/*` Frozen dumps | Kept (referenced by EF-2 / revision); copies under historical archive |
| `results/revision_phase2/`, `phase3c/` | **REVIEW** — not deleted |
| `results/extended_fali/ef5_*`, `ef6_*` | Failed preregistered gates — retained |
| `results/extended_fali/gauge_sensitivity/` (~145 MB local) | Current gauge evidence; hashed in `RAW_DATA_INDEX.json` |
| `tools/elsevier-template/` | Pristine active journal template |
| `.venv/` | Local env — not deleted |

## Current result directories

```text
results/extended_fali/
  ef1_synthetic/
  ef2_gpt2_smoke/
  ef3_two_template/
  ef4_canonical/
  ef5_ablation_robustness/
  ef5_reference_diagnosis/
  ef5_reference_diagnosis_test/
  ef6_pythia_induction/
  gauge_sensitivity/
results/revision_phase1/
results/revision_phase4/
results/CURRENT_PAPER_RESULTS_MANIFEST.json
results/RAW_DATA_INDEX.json
```

## Top 20 largest retained files (excl. `.git`/`.venv`)

See cleanup finalization run; largest are Frozen `results/raw/*.json`, gauge `score_gauge_deltas.pt`, and revision intermediate CSVs (REVIEW).

## Scientific safety

- No unique current-paper evidence deleted.
- EF-5 FAIL and EF-6 FAIL evidence retained.
- Gauge artifacts retained and indexed.
- No `REVIEW` items deleted.


## Tests (cleanup baseline)

Critical subset: **114 passed** in 91.38s (`results/TEST_CLEANUP_BASELINE.txt`).
Full suite: 206 collected; known hangers documented in that file.
