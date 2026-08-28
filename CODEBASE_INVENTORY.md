# Codebase Inventory

Scientifically relevant tracked source (one-line purpose).

## src/

- `src/ioi_dataset.py` — (see file)
- `src/revision_config.py` — Canonical IOI head taxonomy for the FALI revision pipeline.
- `src/revision_dataset.py` — Revision-side IOI dataset validation and metadata helpers.
- `src/revision_factorial.py` — Phase-3C: full-head P × V × C factorial decomposition of conditional_gain.
- `src/revision_fali.py` — Revision FALI: descriptive attention-space Shapley decomposition.
- `src/revision_interventions.py` — Safe ablation hooks for the FALI revision pipeline.
- `src/revision_score_interventions.py` — Clone-safe attention-score / pattern / Q / K interventions for Phase-2/3 revision.
- `src/revision_standard_ioi.py` — Canonical Wang et al. / Redwood Easy-Transformer IOI templates for revision Phase 4.
- `src/revision_stats.py` — Reusable statistics for the FALI revision pipeline.

## experiments/

- `experiments/ablate_primary.py` — (see file)
- `experiments/ablation_controls.py` — experiments/ablation_controls.py
- `experiments/cache_baseline.py` — (see file)
- `experiments/causal_validation.py` — (see file)
- `experiments/fali_decomposition.py` — experiments/fali_decomposition.py
- `experiments/fali_rigorous_validation.py` — fali_rigorous_validation.py
- `experiments/generate_new_figures.py` — generate_new_figures.py
- `experiments/hybrid_intervention.py` — hybrid_intervention.py
- `experiments/output_mediation.py` — experiments/output_mediation.py
- `experiments/qk_decomposition.py` — experiments/qk_decomposition.py
- `experiments/revision_phase1_aggregate.py` — Aggregate Phase-1 full multi-seed results and write scientific summary artifacts.
- `experiments/revision_phase1_conditional_gain.py` — Revision Phase 1: Conditional functional gain of IOI candidate heads
- `experiments/revision_phase1_corrected_classification.py` — Phase-1C: Corrected functional classification from existing Phase-1B raw results.
- `experiments/revision_phase1_parity.py` — Compare CPU smoke vs Modal CUDA smoke per-example quantities.
- `experiments/revision_phase2_aggregate.py` — Phase 2B/3B aggregation: merge Phase-1/2, bootstrap CIs, FALI↔causal,
- `experiments/revision_phase2_fali_causal.py` — Revision Phase 2/3: Descriptive FALI + causal score / pattern / QK interventions.
- `experiments/revision_phase3c_factorial.py` — Phase-3C smoke/full: full-head P×V×C factorial Shapley of conditional_gain.
- `experiments/revision_phase3d_aggregate.py` — Phase-3D aggregation: factorial Shapley + interactions + Phase-2 comparison.
- `experiments/revision_phase4_aggregate.py` — Phase-4B aggregation: broad-template factorial validation.
- `experiments/revision_phase4_factorial.py` — Phase-4A smoke: canonical 30-stratum IOI + Phase-3D P×V×C factorial machinery.
- `experiments/revision_phase4c_recompute.py` — Phase-4C: fix hierarchical bootstrap CIs from existing Phase-4B raw factorial only.
- `experiments/scale_evaluation.py` — scale_evaluation.py
- `experiments/verify_setup.py` — (see file)

## modal/

- `modal_phase1.py` — Modal GPU runner for FALI Revision Phase-1.
- `modal_phase2.py` — Modal GPU runner for FALI Revision Phase 2/3.
- `modal_phase3c.py` — Modal GPU runner for FALI Revision Phase-3C/3D factorial.
- `modal_phase4.py` — Modal GPU runner for FALI Revision Phase-4 canonical-IOI factorial.

## tests/

- `tests/test_build_results_manifests.py` — Tests for tools/build_results_manifests.py.
- `tests/test_revision_phase1.py` — Automated tests for Revision Phase-1 conditional-gain infrastructure.
- `tests/test_revision_phase2.py` — Tests for Revision Phase 2A/3A FALI + score/pattern/Q/K interventions.
- `tests/test_revision_phase3c.py` — Tests for Phase-3C full-head P×V×C factorial decomposition.
- `tests/test_revision_phase4.py` — Tests for Phase-4A canonical IOI expansion.

## tools/

- `tools/capture_baseline_environment.py` — Capture baseline Python environment for reproducibility audits.
- `tools/build_results_manifests.py` — Build and validate RAW_DATA_INDEX.json and HISTORICAL_RESULTS_MANIFEST.json.

## Results / provenance directories

- `results/revision_phase1/` — revision phase outputs (summaries, aggregates, seed runs)
- `results/revision_phase2/` — revision phase outputs (summaries, aggregates, seed runs)
- `results/revision_phase3c/` — revision phase outputs (summaries, aggregates, seed runs)
- `results/revision_phase4/` — revision phase outputs (summaries, aggregates, seed runs)
- `results/raw/` — original-FALI per-example raw CSV/JSON
- `results/summary/` — original-FALI aggregated summaries
- `results/HISTORICAL_RESULTS_MANIFEST.json` — SHA256 manifest of all results files
- `results/RAW_DATA_INDEX.json` — index of large raw files + provenance metadata

## Manuscript / archive

- `paper.tex` — historical manuscript source at repo root (pre-revision thesis)
- `archive/manuscripts/tmlr_rejected_main.tex` — verbatim desk-rejected TMLR submission
- `archive/manuscripts/tmlr_rejected_references.bib` — TMLR-specific bibliography
- `archive/manuscripts/tmlr_rejected_math_commands.tex` — TMLR math macros
- `references.bib` — root bibliography (may differ from TMLR archive)

## Documentation

- `README.md` — current repository overview
- `RESEARCH_STATE.md` — factual research snapshot
- `EXPERIMENT_REGISTRY.md` — completed experiment registry
- `docs/CURRENT_DEFINITIONS.md` — frozen endpoint/factorial definitions
- `docs/LEGACY_README_PRE_REVISION.md` — archived pre-revision README
