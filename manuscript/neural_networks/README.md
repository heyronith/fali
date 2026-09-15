# Neural Networks (Elsevier) — canonical manuscript workspace

**Target journal:** Neural Networks (Elsevier)  
**Entry point:** `main.tex`  
**Class:** `elsarticle` 3.5 (2026-01-09), options `preprint,12pt,authoryear`  
**Bibliography:** `elsarticle-harv.bst` (APA author–year)

## Layout

```text
main.tex              Elsevier shell (placeholders only)
references.bib        Placeholder bib (migrate Factorized bib later)
highlights.tex        Standalone highlights for Editorial Manager
elsarticle.cls        Official class (unmodified copy)
elsarticle-harv.bst   Official Harvard BST (unmodified copy)
figures/              Figure assets (to migrate)
tables/               Optional standalone tables
supplement/           Supplementary materials (to migrate)
JOURNAL_REQUIREMENTS.md
CONTENT_MIGRATION_PLAN.md
COMPILE_REPORT.md
README.md
```

## Build

```bash
export PATH="/Users/ronny/fali-tinytex/bin/universal-darwin:$PATH"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Rules

- Do not edit `elsarticle.cls` or `*.bst`.
- Do not add conference-style vertical compression.
- Do not migrate Frozen Attention Logit Intervention prose as current claims.
- Scientific migration is a separate authorized step.

Pristine template package: `tools/elsevier-template/` (see `SOURCE.md`).
