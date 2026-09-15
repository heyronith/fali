# Machine Learning (Springer) requirements audit

Audit date: 2026-09-15

Primary source: current Machine Learning journal Submission Guidelines and Aims & Scope on Springer Nature Link.

## Required / important items

- Each submission must include a 1-2 page **MLJ Contribution Information Sheet** answering four questions: main claim and importance to ML; precise evidence; closest related papers and relationship; prior publication/overlap and what is new. Incomplete or uninformative sheets can be returned without review.
- Editable source files are required at submission and revision.
- LaTeX is recommended; Springer Nature's current journal article template is encouraged.
- Submit original source (style files and figures included) plus compiled PDF.
- Do **not** use subfolders in the uploaded LaTeX submission package.
- Title should be concise and informative.
- Title page must include authors, affiliations, corresponding-author email, and ORCID if available.
- Abstract: 150-250 words.
- Keywords: 4-6.
- Decimal heading system, no more than three levels.
- Name-year citations; references alphabetized by first-author surname; APA 7 author-list conventions are encouraged; DOI links should be included when available.
- Statements and Declarations must include relevant disclosures; Competing Interests is explicitly required.
- Substantive LLM use should be documented in Methods (or a suitable alternative section); pure AI-assisted copyediting does not require declaration.
- Figures: electronic submission; EPS preferred for vector graphics; fonts embedded; use Fig1/Fig2 naming; lettering normally about 8-12 pt at final size; no titles/captions inside the illustration; panel labels lowercase; captions in manuscript text; large-format target widths 84 mm or 174 mm, max height 234 mm; patterns should supplement color for accessibility.
- Figures should ordinarily appear in the body of the manuscript.
- Supplementary Information should be submitted separately and numbered independently where appropriate.

## Scope fit

Machine Learning explicitly welcomes research-methodology papers that improve how ML research is conducted, but requires substantive, verifiable/replicable evidence and clear positioning within ML. The recommended venue framing is therefore FALI as a **mechanistic-interpretability research methodology for causal mechanism discrimination in learned neural networks**.

## Conversion decisions

- Use the current Factorized-FALI paper as scientific source of truth; do not restore the old Springer manuscript prose.
- Build a fresh `manuscript/machine_learning/` workspace from the current Springer Nature template.
- Retain the current title.
- Use the venue-adapted abstract and Introduction supplied by the research lead.
- Keep Related Work/Discussion largely unchanged, preserving all novelty boundaries.
- Create the mandatory MLJ Contribution Information Sheet from the research-lead draft.
- Replace Elsevier front matter/highlights/graphical-abstract machinery with Springer title page and declarations.
- Move substantive AI-use disclosure into Methods.
- Use the research-lead-generated Springer figure set; no redraw by the editor.
- Flatten final LaTeX package: no figure/bibliography subdirectories in the upload ZIP.
