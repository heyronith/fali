# Baseline File Audit

Audit of scientifically relevant source files at pre-extended-FALI audited baseline.

## Tracked relevant files

Total: **258** files (extensions: py, tex, bib, md, json, yaml, yml, toml).

Representative categories:

- **src/**: 9 files
- **experiments/**: 23 files
- **tests/**: 4 files
- **modal_*.py**: 4 files
- **tools/**: 1 files
- **docs/ + top-level md**: 12 files
- **results/ (selected)**: 20 files

Full tracked list available via `git ls-files`.

## Ignored but present locally (not in main history)

- `.agents/rules/resesrcher.md` — gitignored local/private artifact
- `.pytest_cache/README.md` — gitignored local/private artifact
- `private_camera_ready/acknowledgments.tex` — gitignored local/private artifact
- `private_camera_ready/authors.tex` — gitignored local/private artifact
- `submission_tmlr/README_BUILD.md` — gitignored submission/private copy; archived under archive/manuscripts/
- `submission_tmlr/_backup_pre_lit_fig_20260806/main.tex` — gitignored submission/private copy; archived under archive/manuscripts/
- `submission_tmlr/_backup_pre_lit_fig_20260806/references.bib` — gitignored submission/private copy; archived under archive/manuscripts/
- `submission_tmlr/main.tex` — gitignored submission/private copy; archived under archive/manuscripts/
- `submission_tmlr/math_commands.tex` — gitignored submission/private copy; archived under archive/manuscripts/
- `submission_tmlr/references.bib` — gitignored submission/private copy; archived under archive/manuscripts/

## Untracked relevant files (resolved at `pre-extended-fali-audited`)

The following were committed in the baseline audit repair (HEAD `2c03998`) and are
**no longer untracked**:

- `CODEBASE_INVENTORY.md`
- `archive/manuscripts/*`
- `docs/LEGACY_README_PRE_REVISION.md`
- `tests/test_build_results_manifests.py`
- `tools/capture_baseline_environment.py`

Remaining local-only scientific-adjacent paths are listed under **Ignored** above
(e.g. `submission_tmlr/` originals, `private_camera_ready/`).

## Disposition summary

- Tracked `src/`, `experiments/`, `tests/`, `modal_*.py`, aggregates, and small revision raw CSVs are on main baseline history.
- Large revision raw CSVs remain gitignored on main; minimum set force-added on private branch `audit/raw-baseline` only.
- `submission_tmlr/` remains gitignored; verbatim TMLR source preserved in `archive/manuscripts/`.
- Review-package staging directories (`*_review_package_build/`, patches, zips) remain local-only by design.
