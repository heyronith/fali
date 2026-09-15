# COMPILE REPORT — Neural Networks Elsevier shell

**Workspace:** `manuscript/neural_networks/`  
**Date:** 2026-09-15  
**Result:** **SUCCESS**

## Engine / toolchain

| Tool | Version / path |
|---|---|
| Engine | pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026) via TinyTeX |
| Driver | `pdflatex` |
| Bibliography | `bibtex` + `elsarticle-harv.bst` |
| Build wrapper | `latexmk` 4.88 |
| TinyTeX path | `/Users/ronny/fali-tinytex/bin/universal-darwin` |

## Exact command

```bash
export PATH="/Users/ronny/fali-tinytex/bin/universal-darwin:$PATH"
cd manuscript/neural_networks
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

## Document class

```text
Document Class: elsarticle 2026/01/09, 3.5: Elsevier Ltd
```

Copied unmodified from `tools/elsevier-template/elsarticle/elsarticle.cls`  
(`cmp` identical to pristine extract).

Options used:

```latex
\documentclass[preprint,12pt,authoryear]{elsarticle}
```

Bibliography style: `elsarticle-harv` (APA / author–year; Neural Networks GFA).

## Output

- `main.pdf` produced (4 pages for placeholder shell)
- Exit code: 0
- `latexmk`: All targets up-to-date

## Warnings (non-fatal)

- Hyperref PDF-string warnings from math in abstract placeholder (`$\leq$`)
- Font size substitution warnings from empty graphical-abstract box (template default)
- Overfull `\hbox` on supplement path line (placeholder text)
- Duplicate destination identifiers (hyperref / elsarticle interaction on short shell)

No errors. No missing class/style files.

## Notes

Scientific content not yet migrated. Recompile after prose migration.
