# Experiment Registry — Phase 1A through Phase 4C

Registry of completed revision experiments. Unknown fields marked explicitly.  
Commit SHAs refer to this repository unless noted.

---

## Phase 1A — Smoke / pipeline validation

| Field | Value |
|-------|-------|
| **Phase** | 1A |
| **Purpose** | Validate revision IOI dataset, primary/candidate ablations, attention metrics; smoke N=8 |
| **Script** | `experiments/revision_phase1_conditional_gain.py` |
| **Modal** | `modal_phase1.py` |
| **Dataset** | Historical 2-template `revision_dataset` / `IOIDataset` |
| **N** | 8 (smoke) |
| **Seeds** | 42 |
| **Model** | GPT-2 Small |
| **Hardware / precision** | Modal NVIDIA L4, FP32 (smoke); CPU local tests |
| **Primary intervention** | Zero-ablate candidate head `hook_z`; primary knockout L9H6+L9H9+L10H0 |
| **Candidate heads** | 12 Phase-1 panel |
| **Main outputs** | `results/revision_phase1/smoke/` |
| **Validation** | PASS (`validation_report.md`) |
| **Scientific verdict** | Pipeline validated; proceed to full |
| **Raw data** | `results/revision_phase1/smoke/candidate_raw.csv` (tracked) |
| **Commit SHA** | unknown (pre-baseline local work) |

---

## Phase 1B — Full conditional-gain runs

| Field | Value |
|-------|-------|
| **Phase** | 1B |
| **Purpose** | Full N=200 × 4 seeds; primary-ablation comparison; conditional gain ranking |
| **Script** | `experiments/revision_phase1_conditional_gain.py` |
| **Modal** | `modal_phase1.py` |
| **Dataset** | 2-template mixed A/B |
| **N** | 200 / seed |
| **Seeds** | 42, 1, 8, 22 |
| **Model** | GPT-2 Small |
| **Hardware / precision** | Modal L4, FP32 |
| **Primary intervention** | Corrected primary L9H6+L9H9+L10H0 vs historical B_old |
| **Candidate heads** | 12 |
| **Main outputs** | `results/revision_phase1/full/seed_{seed}/`, `results/revision_phase1/full/aggregate/` |
| **Validation** | PASS all seeds; historical Phase-1 raw untouched checks |
| **Scientific verdict** | Corrected knockout materially different from B_old; recruitment labels assigned |
| **Raw data** | `results/revision_phase1/full/seed_*/candidate_raw.csv` (~637 KB each, tracked) |
| **Commit SHA** | unknown |

---

## Phase 1C — Corrected functional classification

| Field | Value |
|-------|-------|
| **Phase** | 1C |
| **Purpose** | SHA256-stable bootstrap; corrected functional taxonomy; Phase-2 target recommendation |
| **Script** | `experiments/revision_phase1_corrected_classification.py`, `experiments/revision_phase1_aggregate.py` |
| **Dataset** | Phase-1B full outputs |
| **N** | 200 / seed (aggregate) |
| **Seeds** | 42, 1, 8, 22 |
| **Model** | GPT-2 Small (inherited) |
| **Hardware / precision** | CPU aggregation |
| **Primary intervention** | (analysis only) |
| **Candidate heads** | 12 |
| **Main outputs** | `results/revision_phase1/full/aggregate_corrected/` |
| **Validation** | Bootstrap reproducibility; no historical overwrite |
| **Scientific verdict** | L10H1 → suggestive sign-flip; canonical backup set refined |
| **Raw data** | Uses Phase-1B raw |
| **Commit SHA** | unknown |

---

## Phase 2A/3A — FALI + causal score smoke

| Field | Value |
|-------|-------|
| **Phase** | 2A / 3A |
| **Purpose** | Descriptive FALI + causal score hybrids smoke on reporting panel |
| **Script** | `experiments/revision_phase2_fali_causal.py` |
| **Modal** | `modal_phase2.py` |
| **Dataset** | Phase-1 aligned metadata |
| **N** | 8 |
| **Seeds** | 42 |
| **Model** | GPT-2 Small |
| **Hardware / precision** | Modal L4, FP32 |
| **Primary intervention** | Score-row Shapley / pattern / QK patches + FALI |
| **Candidate heads** | Panel subset then extended |
| **Main outputs** | `results/revision_phase2/smoke/` |
| **Validation** | PASS identities |
| **Scientific verdict** | Proceed to full-head causal |
| **Raw data** | `results/revision_phase2/smoke/*_raw.csv` (tracked) |
| **Commit SHA** | unknown |

---

## Phase 2B/3B — Full causal + FALI validation

| Field | Value |
|-------|-------|
| **Phase** | 2B / 3B |
| **Purpose** | Full 12-head causal score + FALI; predictive CV; FALI↔causal correlations |
| **Script** | `experiments/revision_phase2_fali_causal.py`, `experiments/revision_phase2_aggregate.py` |
| **Modal** | `modal_phase2.py` |
| **Dataset** | Phase-1 full metadata per seed |
| **N** | 200 / seed |
| **Seeds** | 42, 1, 8, 22 |
| **Model** | GPT-2 Small |
| **Hardware / precision** | Modal L4, FP32; TL 3.2.1; transformers 4.57.6 |
| **Primary intervention** | Causal score hybrids, pattern/QK patches, descriptive FALI |
| **Candidate heads** | 12 |
| **Main outputs** | `results/revision_phase2/full/aggregate/` |
| **Validation** | PASS all seeds |
| **Scientific verdict** | **PIVOT** — FALI not reliable cross-head causal predictor |
| **Raw data** | Indexed in `results/RAW_DATA_INDEX.json` (large `phase2_combined_raw.csv`, etc.) |
| **Commit SHA** | unknown |

---

## Phase 3C — Factorial smoke (P×V×C)

| Field | Value |
|-------|-------|
| **Phase** | 3C |
| **Purpose** | Full-head 3-factor Shapley smoke; identity validation |
| **Script** | `experiments/revision_phase3c_factorial.py` |
| **Modal** | `modal_phase3c.py` |
| **Dataset** | 2-template IOI |
| **N** | 8 |
| **Seeds** | 42 |
| **Model** | GPT-2 Small |
| **Hardware / precision** | Modal L4, FP32 |
| **Primary intervention** | P×V×C factorial on `hook_z` |
| **Candidate heads** | 12 |
| **Main outputs** | `results/revision_phase3c/smoke/` |
| **Validation** | PASS (endpoint, Shapley, z-recon, L9 invariants) |
| **Scientific verdict** | Proceed to full factorial |
| **Raw data** | `results/revision_phase3c/smoke/factorial_raw.csv` (tracked) |
| **Commit SHA** | unknown |

---

## Phase 3D — Full factorial

| Field | Value |
|-------|-------|
| **Phase** | 3D |
| **Purpose** | Full N=200 × 4 seeds × 12 heads; Phase-2 end-query comparison; aggregate bootstrap |
| **Script** | `experiments/revision_phase3c_factorial.py`, `experiments/revision_phase3d_aggregate.py` |
| **Modal** | `modal_phase3c.py` (mode full-all) |
| **Dataset** | 2-template IOI (100 A + 100 B per seed) |
| **N** | 200 / seed |
| **Seeds** | 42, 1, 8, 22 |
| **Model** | GPT-2 Small |
| **Hardware / precision** | Modal L4, FP32 |
| **Primary intervention** | P×V×C factorial Shapley |
| **Candidate heads** | 12 |
| **Main outputs** | `results/revision_phase3c/full/aggregate/` |
| **Validation** | PASS all seeds |
| **Scientific verdict** | **PARTIAL** (H1–H4 replicate; strict STRONG blocked) |
| **Raw data** | Indexed: `results/revision_phase3c/full/seed_*/factorial_raw.csv` (~1.5 MB each) |
| **Commit SHA** | unknown |

---

## Phase 4A — Canonical IOI smoke

| Field | Value |
|-------|-------|
| **Phase** | 4A |
| **Purpose** | 30-stratum canonical IOI dataset + N=60 factorial smoke |
| **Script** | `experiments/revision_phase4_factorial.py`, `src/revision_standard_ioi.py` |
| **Modal** | `modal_phase4.py` |
| **Dataset** | 15 BABA × ABBA templates; public Easy-Transformer lexicon |
| **N** | 60 (2 × 30 strata) |
| **Seeds** | 42 |
| **Model** | GPT-2 Small |
| **Hardware / precision** | Modal L4, FP32 |
| **Primary intervention** | Phase-3D factorial machinery |
| **Candidate heads** | 12 |
| **Main outputs** | `results/revision_phase4/smoke/` |
| **Validation** | PASS |
| **Scientific verdict** | Execution validated; no science from N=60 |
| **Raw data** | `results/revision_phase4/smoke/factorial_raw.csv` (tracked) |
| **Commit SHA** | unknown |

---

## Phase 4B — Broad-template full validation

| Field | Value |
|-------|-------|
| **Phase** | 4B |
| **Purpose** | Full N=900 × 4 seeds × 30 strata; frozen H1–H4; hierarchical bootstrap |
| **Script** | `experiments/revision_phase4_factorial.py`, `experiments/revision_phase4_aggregate.py` |
| **Modal** | `modal_phase4.py` (mode full-all) |
| **Dataset** | Canonical 30-stratum IOI |
| **N** | 900 / seed (30 examples/stratum) |
| **Seeds** | 42, 1, 8, 22 |
| **Model** | GPT-2 Small |
| **Hardware / precision** | Modal L4, FP32 |
| **Primary intervention** | P×V×C factorial |
| **Candidate heads** | 12 |
| **Main outputs** | `results/revision_phase4/full/aggregate/` |
| **Validation** | PASS all seeds |
| **Scientific verdict** | **BROAD-TEMPLATE PARTIAL** |
| **Raw data** | Indexed: `results/revision_phase4/full/seed_*/factorial_raw.csv` (~6.1 MB each) |
| **Commit SHA** | unknown |

---

## Phase 4C — Bootstrap fix + role audit

| Field | Value |
|-------|-------|
| **Phase** | 4C |
| **Purpose** | Fix hierarchical bootstrap (per-seed independent resampling); recompute CIs; functional-role audit |
| **Script** | `experiments/revision_phase4c_recompute.py` |
| **Modal** | none |
| **Dataset** | Phase-4B raw (unchanged) |
| **N** | 3600 prompts aggregate |
| **Seeds** | 42, 1, 8, 22 |
| **Model** | n/a (reanalysis) |
| **Hardware / precision** | CPU |
| **Primary intervention** | n/a |
| **Candidate heads** | 12 |
| **Main outputs** | `results/revision_phase4/full/aggregate/` (updated CIs), `functional_role_*.csv/md` |
| **Validation** | Tests PASS; means unchanged; no status changes |
| **Scientific verdict** | **BROAD-TEMPLATE PARTIAL** (unchanged) |
| **Raw data** | Unchanged Phase-4B indexed raw |
| **Commit SHA** | recorded at tag `pre-extended-fali-baseline` |

---

## Original FALI experiments (pre-revision)

| Field | Value |
|-------|-------|
| **Phase** | Original |
| **Purpose** | Initial FALI decomposition, QK decomposition, ablations, scale eval |
| **Scripts** | `experiments/fali_decomposition.py`, `experiments/qk_decomposition.py`, etc. |
| **Main outputs** | `results/raw/`, `results/summary/` |
| **Raw data** | Indexed in `results/RAW_DATA_INDEX.json` (large JSON/CSV in `results/raw/`) |
| **Commit SHA** | `cafbcdb` (manuscript cleanup branch); original runs predate revision phases |
