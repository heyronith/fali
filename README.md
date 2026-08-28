# FALI — Functional Attention Circuit Revision Research

Canonical private research repository for the **Frozen Attention Logit Intervention (FALI)** project and its revision pipeline (Phases 1–4). This README describes the **current audited baseline** before any Extended/Causal FALI implementation.

---

## Project status

| Item | State |
|------|-------|
| **Baseline tag (historical snapshot)** | `pre-extended-fali-baseline` — unchanged |
| **Audited baseline tag** | `pre-extended-fali-audited` — documentation + reproducibility fixes |
| **Active branch** | `revision/fali-validity` |
| **Extended / Causal FALI** | Candidate future direction only — **not implemented** |

The original TMLR manuscript thesis (compensation as a general, query-driven algorithmic recruitment story) **failed** as a cross-head explanation under revision Phases 2–4. See [RESEARCH_STATE.md](RESEARCH_STATE.md).

---

## Historical FALI (original project)

**FALI** = **Frozen Attention Logit Intervention** — a pre-softmax target-vs-nontarget counterfactual intervention introduced in the desk-rejected TMLR submission (`archive/manuscripts/tmlr_rejected_main.tex`, also `paper.tex` at repo root).

A later **path-stable FALI analysis** (Phase 2) uses a two-player Shapley decomposition of target attention probability into target-score vs aggregate competing-score contributions. That decomposition is mathematically valid **descriptively** but **did not** reliably predict functional backup recruitment across heads (Phase 2B/3B **PIVOT** verdict).

---

## Revision Phases 1–4 (summary)

| Phase | Focus | Verdict |
|-------|-------|---------|
| **1** | Corrected primary knockout & functional metrics | Four-seed full runs; corrected taxonomy |
| **2** | Causal score interventions + descriptive FALI validation | **PIVOT** — do not treat FALI as general causal explanation |
| **3** | P×V×C factorial Shapley decomposition | **PARTIAL** (2-template IOI) |
| **4** | Canonical 30-strata templates + broad validation | **BROAD-TEMPLATE PARTIAL** (Phase 4C bootstrap fix; no status changes) |

Detailed registry: [EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md)

---

## Current research direction (not implemented)

**Extended / Causal FALI** — structured causal interventions over attention-score computation, value content, and surrounding-network context. This remains a **candidate framework** with no formal definition, proof, or success claims in this repository.

---

## Key documentation

- [RESEARCH_STATE.md](RESEARCH_STATE.md) — factual snapshot through Phase 4C
- [EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md) — scripts, seeds, outputs, verdicts
- [docs/CURRENT_DEFINITIONS.md](docs/CURRENT_DEFINITIONS.md) — frozen endpoint & factorial notation
- [docs/LEGACY_README_PRE_REVISION.md](docs/LEGACY_README_PRE_REVISION.md) — original pre-revision README (historical claims)
- [archive/manuscripts/](archive/manuscripts/) — desk-rejected TMLR source (historical)

**Historical manuscripts:** Root `paper.tex` and `archive/manuscripts/tmlr_rejected_main.tex` are preserved verbatim. Their scientific interpretation was subsequently challenged by revision experiments.

---

## Repository layout

```
src/                  Revision + original IOI / intervention library
experiments/          Original 12-step pipeline + revision phase scripts
modal_phase*.py       Modal GPU runners (not required for aggregation-only work)
tests/                Unit tests for revision pipeline + manifest tooling
results/              Summaries, aggregates, manifests (large raw mostly gitignored)
figures/              Publication figures
tools/                Manifest builder, environment capture
archive/manuscripts/  Historical TMLR submission sources
docs/                 Current definitions + legacy README
```

---

## Tests

```bash
python -m pytest tests/ -q
```

Baseline environment record: `results/TEST_BASELINE_ENVIRONMENT.json`  
Pip freeze: `results/TEST_BASELINE_PIP_FREEZE.txt`

---

## Reproducibility manifests

```bash
python tools/build_results_manifests.py          # rebuild index + manifest
python tools/build_results_manifests.py --validate
```

Raw audit branch (private canonical repo only): `audit/raw-baseline` — minimum critical raw CSVs for independent verification.

---

## Legacy public mirror

The earlier public repository [`heyronith/attention-repairs-itself`](https://github.com/heyronith/attention-repairs-itself) is preserved as remote `legacy-public`. Canonical development uses [`heyronith/fali`](https://github.com/heyronith/fali) (private).
