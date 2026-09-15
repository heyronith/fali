# TEMPLATE AUDIT BASELINE

**Created:** 2026-09-15  
**Audit branch:** `manuscript/neural-networks-template-audit`  
**STOP CASE:** Neural Networks / Elsevier template search (documented after inventory)

## Repository identity

| Field | Value |
|---|---|
| Owner/name | `heyronith/fali` |
| Repository root | `/Users/ronny/Desktop/Research/AI ALIGNMENT/FALI` |
| Remote URL (`origin`) | `https://github.com/heyronith/fali.git` |
| Also configured | `legacy-public` → `https://github.com/heyronith/mechanistic-intepretability.git` |
| Source branch | `manuscript/reconstruction` |
| Source HEAD SHA | `9672c029ad64dcce746b103125ff5f4e6619f842` |
| Source tip message | `docs: correct author contact metadata` |
| Audit branch | `manuscript/neural-networks-template-audit` (created from source HEAD; not merged) |

## Checkout procedure used

```bash
git checkout manuscript/reconstruction
git pull --ff-only origin manuscript/reconstruction   # Already up to date
git checkout -b manuscript/neural-networks-template-audit
```

`manuscript/reconstruction` exists locally and on `origin`.

## Git status at audit start (after branch creation)

Working tree relative to `9672c02`:

- **Tracked modified:** none (clean relative to HEAD for tracked files)
- **Untracked (selected):**
  - `fali_manuscript_integration.zip`, `fali_manuscript_integration/`
  - `fali_submission_finalization_package.zip`
  - `manuscript/machine_learning.zip`
  - `manuscript/machine_learning/*.{aux,bbl,blg,fdb_latexmk,fls,log,out}` and `review_renders/`
  - `results/extended_fali/ef5_reference_diagnosis_test/`
  - `results/extended_fali/gauge_sensitivity/` (present locally; from other branch work; not on this HEAD)
  - `tools/install-tinytex.sh`, `tools/pydeps/`, `tools/springer-template/`, `tools/tectonic.tar.gz`, `tools/texlive-local/`

## Ignored / build artifacts under manuscript

LaTeX aux/log/out and `review_renders/` under `manuscript/machine_learning/` are untracked build products (not the scientific source of truth).

## Case C note (preview)

No `elsarticle.cls`, `cas-sc.cls`, `cas-dc.cls`, or Elsevier Neural Networks package was found in the repository, untracked project trees searched, or project zip inventories.  
See `TEMPLATE_INVENTORY.md` and the lead response: **NEURAL NETWORKS TEMPLATE NOT FOUND**.

Destructive cleanup and creation of a fake `manuscript/neural_networks/` shell were **not** performed.
