# TEMPLATE INVENTORY

**Branch:** `manuscript/neural-networks-template-audit`  
**Source HEAD:** `9672c029ad64dcce746b103125ff5f4e6619f842`  
**Verdict:** **NEURAL NETWORKS TEMPLATE NOT FOUND**

Search covered tracked sources, common untracked project dirs, and zip listings (`*.zip` at repo root / `manuscript/` / `tools/springer-template/`). No Elsevier `elsarticle` / CAS template package is present.

---

## Summary table

| Path | Tracked? | `\documentclass` / family | Venue | FALI scientific prose? | Role |
|---|---|---|---|---|---|
| `manuscript/machine_learning/main.tex` | tracked | `sn-jnl` `[pdflatex,sn-mathphys-ay]` | Springer / Machine Learning | YES — current Factorized FALI | **ACTIVE** manuscript entry |
| `manuscript/machine_learning/supplement.tex` | tracked | `sn-jnl` | Springer / Machine Learning | YES | ACTIVE supplement |
| `manuscript/machine_learning/contribution_information_sheet.tex` | tracked | `article` 11pt | generic / Springer submission aid | YES (metadata) | ACTIVE support |
| `manuscript/machine_learning/sn-jnl.cls` | tracked | Springer Nature journal class | Springer | N/A (class) | ACTIVE template support |
| `manuscript/machine_learning/sn-mathphys-ay.bst` | tracked | BST | Springer | N/A | ACTIVE bib style |
| `tools/springer-template/extracted/sn-article-template/` | untracked | official SN package Dec 2024 | Springer | NO (boilerplate) | Template cache / extract |
| `tools/springer-template/sn-journal-template.zip` | untracked | zip of SN package | Springer | NO | Downloaded Springer package |
| `archive/manuscripts/tmlr_rejected_main.tex` | tracked | `article` + `\usepackage{tmlr}` | TMLR | YES — Frozen Attention Logit Intervention | HISTORICAL |
| `submission_tmlr/main.tex` | tracked (dir) | `article` + `tmlr` | TMLR | YES — historical | HISTORICAL / duplicate workspace |
| `submission_tmlr/tmlr.sty`, `tmlr.bst` | tracked | TMLR style | TMLR | N/A | HISTORICAL template support |
| `tmlr-style-file-main/` | tracked/untracked mix | upstream TMLR style package | TMLR | example only | TEMPLATE ONLY |
| `paper.tex` | tracked | `article` 12pt a4 | generic article | YES — Frozen Attention Logit Intervention | HISTORICAL root draft |
| Elsevier `elsarticle*` / `cas-*` | **ABSENT** | — | Neural Networks / Elsevier | — | **NOT FOUND** |

---

## Candidates in detail

### 1. `manuscript/machine_learning/` — ACTIVE Springer Machine Learning

- **Document class:** `\documentclass[pdflatex,sn-mathphys-ay]{sn-jnl}`
- **Bibliography:** `sn-mathphys-ay` / `references.bib`
- **Front matter:** Springer `\title`, `\author`, `\institute`, `\abstract`, `\keywords`
- **Venue evidence:** `README.md` explicitly names Machine Learning (Springer Nature; journal 10994); `COMPILE_AUDIT.md` records Dec 2024 SN package SHA256 for `sn-jnl.cls`
- **Scientific content:** Factorized Attention-Level Intervention (current FALI); split inputs `method_validation.tex`, `gpt2_ioi.tex`, `pythia_induction.tex`, `discussion_limitations_conclusion.tex`
- **Last commit touching `main.tex`:** `9672c02` (`docs: correct author contact metadata`)
- **Status:** ACTIVE — this is what recent manuscript work has been editing

### 2. `archive/manuscripts/tmlr_rejected_*` — HISTORICAL TMLR

- **Class:** `\documentclass[10pt]{article}` + `\usepackage{tmlr}`
- **Title theme:** “When Attention Repairs Itself…” / Frozen Attention Logit Intervention
- **Status:** HISTORICAL (already under `archive/manuscripts/`)

### 3. `submission_tmlr/` — HISTORICAL TMLR submission tree

- Full TMLR build tree (`main.tex`, `tmlr.sty`, `tmlr.bst`, figures, PDF)
- Near-duplicate of archived rejected manuscript
- **Status:** HISTORICAL / should not look “active” for Neural Networks migration

### 4. `tmlr-style-file-main/` — upstream TMLR style package

- Contains example `main.tex`, `tmlr.sty`, `tmlr.bst`, LICENSE
- **Status:** TEMPLATE ONLY

### 5. `paper.tex` (repo root)

- **Class:** `\documentclass[12pt,a4paper]{article}`
- Generic Times/`natbib` article; **not** TMLR/Springer/Elsevier class
- Content: Frozen Attention Logit Intervention
- **Status:** HISTORICAL generic draft

### 6. Elsevier / Neural Networks

Searched for: `elsarticle.cls`, `elsarticle-template*.tex`, `cas-sc.cls`, `cas-dc.cls`, `cas-*.tex`, paths/names containing `neural_network(s)`, `elsevier`, zip members matching `elsarticle|cas-|elsevier|neural`.

**Result:** none.

Also checked zip inventories of `fali_manuscript_integration.zip`, `fali_submission_finalization_package.zip`, `manuscript/machine_learning.zip`, `submission_tmlr.zip`, `tools/springer-template/sn-journal-template.zip` — no Elsevier template members.

---

## Support files (`.cls` / `.sty` / `.bst`)

| Path | Type | Venue |
|---|---|---|
| `manuscript/machine_learning/sn-jnl.cls` | cls | Springer |
| `manuscript/machine_learning/sn-mathphys-ay.bst` | bst | Springer |
| `tools/springer-template/extracted/sn-article-template/sn-jnl.cls` | cls | Springer (untracked extract) |
| `tools/springer-template/extracted/sn-article-template/bst/*.bst` | bst | Springer |
| `submission_tmlr/tmlr.sty`, `submission_tmlr/tmlr.bst` | sty/bst | TMLR |
| `submission_tmlr/fancyhdr.sty` | sty | TMLR dependency |
| `tmlr-style-file-main/tmlr.sty`, `tmlr.bst`, `fancyhdr.sty` | sty/bst | TMLR |

**No** `elsarticle.cls` / CAS class files anywhere in the project tree searched.

---

## Case classification

**Case C — no Neural Networks template found.**

Per research-lead instructions: destructive cleanup stopped; no directory was renamed to pretend it is Neural Networks; repository scientific sources preserved; only audit documentation added on this branch.
