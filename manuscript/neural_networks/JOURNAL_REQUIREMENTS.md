# JOURNAL REQUIREMENTS — Neural Networks (Elsevier)

Sources consulted (2026-09-15):

1. Neural Networks Guide for Authors  
   https://www.sciencedirect.com/journal/neural-networks/publish/guide-for-authors
2. Elsevier LaTeX instructions  
   https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions
3. Official `elsarticle` package README / templates (v3.5, 2026-01-09)  
   https://ctan.org/pkg/elsarticle

No journal-specific class other than the standard Elsevier `elsarticle` pathway was mandated.

| Requirement | Official source | Mandatory / recommended | How this manuscript will satisfy it |
|---|---|---|---|
| Article type: Original full-length article (or Letter ≤2500 words) | Neural Networks GFA | Choose at submission | Full-length research article (Factorized FALI) |
| Section classification (Cognitive Science / Neuroscience / Learning Systems / Mathematical and Computational Analysis / Engineering and Applications) | Neural Networks GFA | Mandatory at submission | Select at Editorial Manager; likely Learning Systems or Mathematical and Computational Analysis — verify before submit |
| Editable LaTeX source (`.tex`), not PDF-only | Neural Networks GFA; Elsevier LaTeX | Mandatory | Provide `main.tex` + inputs + class/bst |
| Use Elsevier LaTeX template (`elsarticle`) | Neural Networks GFA (encourages); Elsevier LaTeX | Recommended / expected | `\documentclass[preprint,12pt,authoryear]{elsarticle}` |
| Title page with authors, affiliations, corresponding author | GFA | Mandatory | `frontmatter` author/affiliation/`\cortext` placeholders |
| Abstract ≤ 250 words, standalone | GFA | Mandatory | Abstract environment; migrate Factorized abstract later |
| Keywords: 1–6 | GFA | Mandatory | `keyword` environment (placeholder set) |
| Highlights: 3–5 bullets, ≤85 chars each; separate editable file with “highlights” in name | GFA | Encouraged | `highlights` in `main.tex` + standalone `highlights.tex` |
| Graphical abstract | Elsevier template provides hook; GFA artwork rules | Optional unless portal requires | Empty `graphicalabstract` reserved; do not invent |
| Numbered sections / subsections | GFA | Mandatory | Standard `\section` / `\subsection` |
| APA 7th author–year references; alphabetical list | GFA | Mandatory | `authoryear` + `elsarticle-harv.bst` |
| Acknowledgements before references | GFA | If applicable | Dedicated unnumbered section |
| Funding sources | GFA | Mandatory disclosure | Dedicated section placeholder |
| CRediT author contributions | GFA | Required for corresponding authors | Dedicated section placeholder |
| Competing interests declaration | GFA | Mandatory | Dedicated section placeholder |
| Research data / code availability | GFA research-data policy | Required statement | Dedicated section placeholder |
| Generative AI disclosure (if used beyond grammar tools) | GFA | Conditional | Add section only if applicable |
| Supplementary material | GFA | Encouraged when relevant | `supplement/` + Appendix pointer |
| Figures as separate files; cite in order | GFA | Mandatory for artwork | `figures/` directory |
| Source-file submission via Editorial Manager | GFA / Elsevier LaTeX | Mandatory | Zip of tex/bib/cls/bst/figures |

## Reference style evidence

Neural Networks GFA: “Citations in the text should follow the referencing style used by the American Psychological Association” and “reference list should be arranged alphabetically and then chronologically.”

Therefore: **Harvard / author–year** via official `elsarticle-template-harv.tex` pattern and `elsarticle-harv.bst` (not numerical).

## Class specificity conclusion

Neural Networks does **not** require a custom journal class beyond Elsevier’s standard LaTeX (`elsarticle`). APA author–year bibliography and GFA declarations/highlights are the main journal-specific constraints beyond the generic Elsevier shell.
