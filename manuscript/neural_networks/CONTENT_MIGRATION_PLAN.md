# CONTENT MIGRATION PLAN — Neural Networks

**Active shell:** `manuscript/neural_networks/`  
**Do not migrate prose in this phase.**  
**Scientific source of truth for Factorized FALI:** `manuscript/archive/machine_learning/`  
**Source commit for Factorized prose (pre-archive HEAD on audit line):** `a32cb2cc6b108f4b30e3ddd02824ddb53ff2c8b9` (content identical to `9672c02` Factorized files; archive move preserves blobs)

Frozen Attention Logit Intervention sources (`paper.tex`, `archive/manuscripts/tmlr_*`, `submission_tmlr/`) are **HISTORICAL** and must never be silently migrated as current claims.

| Section | Source path | Source commit | Current scientific status | Safe to migrate verbatim? | Requires revision? | Reason |
|---|---|---|---|---|---|---|
| Title | `manuscript/archive/machine_learning/main.tex` | `9672c02` / `a32cb2c` | CURRENT Factorized | Mostly | Light front-matter adapt | Elsevier `\title` / `\tnotetext` |
| Abstract | same | same | CURRENT | Mostly | Check ≤250 words | GFA abstract limit |
| Keywords | same / new | same | CURRENT | Partial | Yes | Map to 1–6 Elsevier keywords |
| Highlights | none finalized | — | PLACEHOLDER only | No | Yes | Write 3–5 × ≤85 chars |
| Introduction | `.../main.tex` | same | CURRENT | Mostly | Venue copy-edit only | Supersession note vs Frozen is CURRENT |
| Related Work | `.../main.tex` | same | CURRENT | Mostly | Review | Embedded in main |
| Problem Formulation | `.../main.tex` | same | CURRENT | Mostly | No scientific rewrite | |
| FALI Method | `.../main.tex` § Factorized… | same | CURRENT | Mostly | No | Factorized only |
| Method Validation | `.../method_validation.tex` | same | CURRENT | Yes | Minimal | `\input` → section body |
| GPT-2 IOI | `.../gpt2_ioi.tex` | same | CURRENT | Yes | Minimal | |
| EF-5 robustness | ledgers + `results/extended_fali/` EF-5; limited dedicated `.tex` | `9672c02` tree | CURRENT experiments; prose sparse | Partial | Yes | May need a short section from evidence ledgers — no guessing numbers |
| Gauge-sensitivity analysis | **Results** on branch `experiment/gauge-sensitivity` @ `43135198065e4a13acb241c677a0965e2165b988`; report `results/extended_fali/gauge_sensitivity/GAUGE_SENSITIVITY_REPORT.md` (local/unmerged). **Manuscript prose:** not present on `manuscript/reconstruction` / audit HEAD | see below | CURRENT SCIENCE (analysis done); prose NOT in active paper | No (results≠prose) | Yes | New authorized section after scientific review of audit A |
| Pythia induction | `.../pythia_induction.tex` | `9672c02` | CURRENT | Yes | Minimal | |
| Discussion | `.../discussion_limitations_conclusion.tex` | same | CURRENT | Yes | Minimal | |
| Limitations | same file | same | CURRENT | Yes | Minimal | |
| Conclusion | same file | same | CURRENT | Yes | Minimal | |
| Bibliography | `.../references.bib` | same | CURRENT | Yes | Restyle to APA/`elsarticle-harv` | Do not use TMLR bib as primary |
| Figures | `.../figures/` | same | CURRENT | Yes (assets) | Wrapper adapt | Do not regenerate |
| Supplement | `.../supplement.tex` | same | CURRENT | Yes | Elsevier supplement packaging | |
| Author metadata / CRediT | `.../main.tex` + contribution sheet | `9672c02` | CURRENT | Partial | Yes | Map to Elsevier frontmatter + CRediT section |

## Gauge-analysis location (explicit)

| Field | Value |
|---|---|
| Branch | `experiment/gauge-sensitivity` |
| Tip commit (results) | `43135198065e4a13acb241c677a0965e2165b988` (`exp: add preregistered FALI gauge sensitivity results`) |
| Preregistration commit | `e60f81acfff8fc9dd4f924ebb1c087ec89734c05` |
| Result directory | `results/extended_fali/gauge_sensitivity/` |
| Key report | `GAUGE_SENSITIVITY_REPORT.md` (+ CSV/JSON summaries) |
| Manuscript prose on audited reconstruction HEAD | **Not found** — flag: requires authorized drafting after scientific review |
| Blind merge into this branch? | **No** |

## Historical sources (do not migrate as current)

- `paper.tex`
- `archive/manuscripts/tmlr_rejected_main.tex`
- `submission_tmlr/main.tex`
