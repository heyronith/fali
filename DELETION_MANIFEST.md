# DELETION MANIFEST

**Branch:** `maintenance/current-fali-cleanup`  
**Baseline HEAD:** `bf6f89e470474a5be4def6dfa76ceada549224b0`  
**Safety tag:** `pre-current-fali-cleanup`  
**Policy:** Commit this manifest BEFORE destructive cleanup. Do not delete `REVIEW` items.

Inventory source: `PROJECT_FILE_INVENTORY.csv`

---

## Files/directories to DELETE

Evidence standard for each: not referenced by current manuscript AND not required by current experiment code AND not required to reproduce current figures/results.

### Review / audit packages (generated)

| Path | Approx size | Reason |
|---|---|---|
| `fali_review_package.zip` | ~1.1 MB | Generated review bundle; contents obsolete |
| `fali_manuscript_integration.zip` | ~1.2 MB | Duplicate manuscript export |
| `fali_manuscript_integration/` | ~1.5 MB | Unpacked export |
| `fali_submission_finalization_package.zip` | ~0.8 MB | Old finalization bundle |
| `phase1a_review_package.zip` | | Phase review ZIP |
| `phase1a_review_package_build/` | | Build staging |
| `phase1a_review_staging/` | | Staging |
| `phase1b_review_package.zip` | ~1.2 MB | Phase review ZIP |
| `phase1b_review_build/` | ~2.9 MB | Build staging |
| `phase1c_review_package.zip` | | Phase review ZIP |
| `phase1c_review_build/` | | Build staging |
| `phase2a3a_review_package.zip` | | Phase review ZIP |
| `phase2a3a_review_package_build/` | | Build staging |
| `phase2b3b_review_package.zip` | ~3.6 MB | Phase review ZIP |
| `phase2b3b_review_package_build/` | ~9.2 MB | Build staging |
| `phase3c_review_package.zip` | | Phase review ZIP |
| `phase3c_review_package_build/` (if present) | | Build staging |
| `phase3d_review_package.zip` | ~2.2 MB | Phase review ZIP |
| `phase3d_review_package_build/` | ~6.7 MB | Build staging |
| `phase4a_review_package.zip` | | Phase review ZIP |
| `phase4a_review_package_build/` (if present) | | Build staging |
| `phase4b_review_package.zip` | | Phase review ZIP |
| `phase4b_review_package_build/` (if present) | | Build staging |
| `phase4c_review_package.zip` | | Phase review ZIP |
| `phase4c_review_package_build/` (if present) | | Build staging |
| `review_package/` (if present) | | Legacy review tree |
| `git_diff_phase*.patch` | | Historical patch exports |
| `PHASE*_REVIEW_SUMMARY.md` | small | Optional: **KEEP** process summaries? → **DELETE** duplicates of registry; registry retained |

### Venue / manuscript clutter

| Path | Reason |
|---|---|
| `submission_tmlr/` | Entire TMLR working tree + tectonic caches (~51 MB); one archival `.tex` copied first |
| `submission_tmlr.zip` | Packaged TMLR submission |
| `tmlr-style-file-main/` | Unused TMLR style clone |
| `manuscript/machine_learning.zip` | Duplicate of archived Factorized tree |
| `manuscript/Untitled` | Scratch |
| `Hydra.pdf`, `IOI.pdf` | Unrelated downloaded PDFs |
| `figures/` (repo root) | Superseded by `manuscript/neural_networks/figures/` |

### Local tooling extracts (not pristine Elsevier)

| Path | Reason |
|---|---|
| `tools/texlive-local/` | Local TeX binary extract (~52 MB) |
| `tools/tectonic.tar.gz` | Local TeX tarball (~20 MB) |
| `tools/pydeps/` | Vendored pypdf extract (~6 MB) |
| `tools/springer-template/` | Springer extract (active venue is Elsevier; pristine Springer not required) |
| `tools/install-tinytex.sh` | Local helper; TinyTeX lives outside repo |

**KEEP:** `tools/elsevier-template/` (including `elsarticle.zip`).

### Caches / temps

| Path | Reason |
|---|---|
| `__pycache__/` (all) | Regenerable |
| `.pytest_cache/` | Regenerable |
| `*.pyc` | Regenerable |
| `.DS_Store` | OS junk |
| `manuscript/**/*.{aux,bbl,blg,fls,fdb_latexmk,synctex.gz,spl,out}` except intentional | LaTeX temps |
| `manuscript/**/*.log` build logs | LaTeX temps |
| `submission_tmlr` LaTeX/tectonic caches (covered by deleting tree) | |

### Compiled PDFs (stale non-current)

| Path | Reason |
|---|---|
| Historical manuscript PDFs under `submission_tmlr/` | Deleted with tree |
| Archive ML aux/PDF rebuilds if present as temps | Clean aux; **KEEP** `manuscript/archive/machine_learning/main.pdf` if present as provenance render |

**KEEP current:** `manuscript/neural_networks/main.pdf`, `manuscript/neural_networks/supplement/supplement.pdf`

---

## Files/directories to ARCHIVE

Destination root: `archive/historical_fali/`

| Source | Destination | Reason |
|---|---|---|
| `paper.tex` | `archive/historical_fali/paper_frozen_attention.tex` | Frozen Attention manuscript; cited as superseded in Related Work |
| `submission_tmlr/paper.tex` or primary TMLR tex (if distinct) | `archive/historical_fali/tmlr_submission_main.tex` | One authoritative TMLR copy (skip if identical to existing `archive/manuscripts/tmlr_rejected_main.tex`) |
| `results/raw/*` (Frozen-era dumps) | **copy** → `archive/historical_fali/raw/` (originals **KEEP**) | Referenced by EF-2 / revision scripts; do not remove from `results/raw/` |
| `experiments/qk_decomposition.py` | `archive/historical_fali/experiments/` | Frozen-only script |
| `experiments/output_mediation.py` | `archive/historical_fali/experiments/` | Frozen-only script |
| `experiments/fali_decomposition.py` | `archive/historical_fali/experiments/` | Attention-probability FALI era |
| `experiments/fali_rigorous_validation.py` | `archive/historical_fali/experiments/` | Superseded validation |
| `experiments/hybrid_intervention.py` | `archive/historical_fali/experiments/` | Historical |
| `experiments/ablate_primary.py` | `archive/historical_fali/experiments/` | Historical (if unused by current) |
| `experiments/cache_baseline.py` | `archive/historical_fali/experiments/` | Historical |
| `experiments/causal_validation.py` | `archive/historical_fali/experiments/` | Historical |
| `experiments/scale_evaluation.py` | `archive/historical_fali/experiments/` | Historical |
| `experiments/generate_new_figures.py` | `archive/historical_fali/experiments/` | Superseded root figure generator |
| `experiments/ablation_controls.py` | `archive/historical_fali/experiments/` | Historical controls |
| `docs/LEGACY_README_PRE_REVISION.md` | keep in place OR copy note into `archive/historical_fali/README.md` | Provenance |

Also write `archive/historical_fali/README.md` describing Frozen vs Factorized supersession.

Existing `archive/manuscripts/tmlr_rejected_*` retained (already minimal).

---

## Files retained despite being old

| Path | Reason |
|---|---|
| `results/extended_fali/ef5_*` including FAIL evidence | Paper discusses EF-5 overall gate failure |
| `results/extended_fali/ef6_*` FAIL transfer | Paper’s central negative result |
| `results/extended_fali/ef3_two_template/` | Referenced in GPT-2 narrative as earlier screen |
| `results/revision_phase1/` | Corrected recruitment / attention dissociation |
| `results/revision_phase4/` | Supplement / figure provenance pointer |
| `results/extended_fali/gauge_sensitivity/` (~145 MB) | Current gauge analysis; **KEEP** (local raw gz + committed summaries) |
| `manuscript/archive/machine_learning/` | Factorized Springer scientific source |
| `tools/elsevier-template/` | Pristine active journal template |
| `docs/EXTENDED_FALI_*`, EF plans | Preregistration / formalization |
| `EXPERIMENT_REGISTRY.md`, `RESEARCH_STATE.md` | Process provenance |
| `.venv/` | Local environment — **not deleted** (gitignored) |

---

## REVIEW items (NO DELETE in this cleanup)

| Path | Why REVIEW |
|---|---|
| `results/revision_phase2/` | Superseded exploratory Factorized revision; may still hold intermediate provenance |
| `results/revision_phase3c/` / `phase3d` | Same |
| `experiments/revision_phase2_*.py`, `revision_phase3*.py` | Scripts for REVIEW results |
| `results/raw/fali_raw_results.*` | Ambiguous naming; not clearly Frozen-only |
| `.agents/`, IDE metadata | Non-scientific; leave |
| Large ignored cell_raw under EF trees | Already gitignored; required for recompute — **KEEP locally** |

---

## Explicit non-actions (scientific safety)

- Do **not** delete failed EF-5 / EF-6 evidence.
- Do **not** delete gauge artifacts unless archived with hashes (prefer preserve).
- Do **not** sanitize unfavorable results.
- Do **not** delete `REVIEW` rows.
- Do **not** delete `.venv`.
