# RESEARCH LEAD INSTRUCTION — MACHINE LEARNING (SPRINGER) CONVERSION

Repository: `heyronith/fali`

Scientific source branch: `manuscript/neural-networks-production`

Required source HEAD: `95358add53bee019d9ae901a7305d7566ad6676f`

Create branch: `manuscript/machine-learning-conversion`

## Inputs supplied by research lead

The user will provide the research-lead package `FALI_MLJ_Transition_Package.zip`. Treat the following files inside it as authoritative:

- `MLJ_REQUIREMENTS_AUDIT.md`
- `MLJ_SCIENTIFIC_ADAPTATION.md`
- `MLJ_CONTRIBUTION_INFORMATION_SHEET.md`
- `MLJ_FIGURE_CAPTIONS.tex`
- `generate_mlj_figures.py`
- `figure_data.json`
- `FIGURE_MANIFEST.json`
- `figures/Fig1.{pdf,eps,svg,png}` through `figures/Fig5.{pdf,eps,svg,png}`

**Do not redraw, restyle, or numerically modify the figures.** Figure generation is research-lead controlled. Use the supplied PDF/EPS outputs exactly. You may change only LaTeX sizing/placement. If the supplied figure files are not locally available, STOP and request them rather than regenerating replacements.

## Goal

Convert the final Factorized-FALI paper from the Elsevier Neural Networks production workspace into a complete, submission-ready package for **Machine Learning (Springer Nature)**, preserving the approved science and using the research-lead-generated Springer figures.

This is a venue conversion, not a new scientific rewrite.

---

## 1. Verify current Machine Learning requirements before editing

Use the current official Machine Learning submission guidelines and Springer Nature LaTeX author support. Confirm and document:

- mandatory 1–2 page MLJ Contribution Information Sheet;
- current Springer Nature LaTeX template version;
- abstract 150–250 words;
- 4–6 keywords;
- title-page author/affiliation/corresponding-author/ORCID fields;
- Statements and Declarations requirements;
- author–year reference style and journal's APA 7 guidance;
- figure format/lettering/placement requirements;
- no-subfolder rule for LaTeX submission upload;
- Supplementary Information requirements;
- LLM disclosure placement in Methods.

Create:

`manuscript/machine_learning/MLJ_REQUIREMENTS_VERIFICATION.md`

Do not infer the bibliography style solely from an old FALI Springer draft. Verify against the current official `sn-jnl` package. Prefer an official Springer style only. If the current template's APA-compatible option is `sn-apacite`, verify rendered output before selecting it. Do not modify `.bst` files manually.

---

## 2. Build a fresh Machine Learning workspace

Create:

`manuscript/machine_learning/`

Use the current official Springer Nature journal-article LaTeX template (`sn-jnl` or its current official successor).

Do NOT reactivate old archived Machine Learning prose as the scientific source. Old Springer files may be consulted only for provenance/template history.

Scientific content source of truth:

`manuscript/neural_networks/`

at the required source HEAD plus the research-lead `MLJ_SCIENTIFIC_ADAPTATION.md`.

---

## 3. Scientific text migration

### Title

Use exactly:

**FALI: Mechanistic Decomposition of Conditional Attention-Head Causal Usefulness**

### Abstract

Replace the Elsevier abstract with the venue-adapted abstract in `MLJ_SCIENTIFIC_ADAPTATION.md`.

### Keywords

Use exactly six:

- mechanistic interpretability
- causal intervention
- transformer circuits
- attention heads
- conditional importance
- causal attribution

### Introduction

Replace the current Introduction with the research-lead `Replacement Introduction` from `MLJ_SCIENTIFIC_ADAPTATION.md`.

This replacement is authoritative. Do not independently paraphrase it.

### Related Work / Problem Formulation / Method / Validation / Results / Robustness / Pythia / Discussion / Limitations / Conclusion

Migrate the approved scientific content from the current production manuscript with venue-neutral formatting only.

Do not restore old Frozen-FALI interpretations or old Springer prose.

Preserve the current research-lead narrative:

1. conditional importance detects THAT usefulness changed;
2. FALI asks WHY it changed;
3. GPT-2 L10H2 is positively recruited; L10H10 is not;
4. GPT-2 canonical fine mechanism is `T>0, R<0`;
5. Pythia positive recruitment recurs but fine mechanism reverses `T<0, R>0`;
6. the preregistered mechanism-transfer hypothesis failed;
7. shared high-level function does not imply shared fine mechanism.

---

## 4. Prior-work transparency

The Introduction and Related Work must both preserve explicit disclosure of the earlier public **Frozen Attention Logit Intervention** preprint.

Do not describe the present paper as simply a renamed version.

Preserve the substantive differences specified by the research lead:

- attention-probability estimand → output-grounded causal usefulness;
- target/background decomposition → T/R/O/V/C;
- standard Owen + coarse/fine reconciliation;
- controlled validation;
- corrected GPT-2 recruitment interpretation;
- broad canonical confirmation;
- matched-ablation robustness with failed overall gate retained;
- gauge sensitivity;
- Pythia mechanism-transfer test and failed transfer prediction.

---

## 5. Mandatory MLJ Contribution Information Sheet

Create a standalone 1–2 page document from:

`MLJ_CONTRIBUTION_INFORMATION_SHEET.md`

Final deliverables:

`FALI_MLJ_Contribution_Information_Sheet.tex`

`FALI_MLJ_Contribution_Information_Sheet.pdf`

Preserve all four question headings required by MLJ.

Before finalization, flag the line asking authors to confirm whether any additional conference/workshop/journal version overlaps beyond the cited public preprint. Do not silently delete this author-confirmation requirement.

Do not pad the sheet with generic claims; keep the precise empirical values and closest-work distinctions supplied by the research lead.

---

## 6. Springer front matter

Convert Elsevier-specific front matter to Springer syntax.

Remove:

- `elsarticle` class;
- `\journal{Neural Networks}`;
- Elsevier graphical-abstract environment;
- Elsevier Highlights environment/files from the active manuscript;
- Elsevier-specific CRediT/front-matter commands;
- Elsevier bibliography style.

Retain verified author names, affiliations, emails, and corresponding-author designation.

Add ORCID fields only if known. Do not invent them.

Flag postal information if the template/journal requires details not currently verified.

---

## 7. LLM disclosure

Springer guidance requires substantive LLM use to be documented in Methods or a suitable alternative section.

Remove the Elsevier standalone generative-AI declaration and insert the exact research-lead paragraph from `MLJ_SCIENTIFIC_ADAPTATION.md` near the end of the Methods section.

Do not weaken or expand the disclosure.

---

## 8. Statements and Declarations

Create Springer-style `Statements and Declarations` with, at minimum:

- Funding
- Competing Interests
- Data and Code Availability
- Author Contributions (converted from the existing verified contribution statement)

Use research-lead wording where supplied.

Do not invent a public DOI/repository URL. If no persistent archive exists yet, retain the accurate future-release wording and flag it as a submission-time item.

---

## 9. Figures — USE RESEARCH-LEAD FILES ONLY

Replace all old manuscript figures with the supplied Springer figure set:

- `Fig1.pdf`
- `Fig2.pdf`
- `Fig3.pdf`
- `Fig4.pdf`
- `Fig5.pdf`

The research lead has already regenerated these from verified values at Springer large-format width (174 mm target), with 8–12 pt-scale lettering, lowercase panel labels, external captions, and redundant marker/hatch encodings for accessibility.

### Important structural change

Use **one combined Fig4**. Remove the Elsevier split `Fig4a` / `Fig4bc` continuation machinery from the active Machine Learning manuscript.

Use captions from:

`MLJ_FIGURE_CAPTIONS.tex`

Do not insert titles/captions into the artwork.

Use PDF for `pdflatex` compilation. Retain EPS as the preferred editable vector submission artifact. Retain SVG/source script/data in the reproducibility package, not necessarily as active LaTeX inputs.

Verify each supplied figure against `FIGURE_MANIFEST.json` before packaging. Any hash mismatch must be reported.

---

## 10. References

Machine Learning uses name–year citation style and encourages APA 7 author-list conventions.

Audit every active bibliography entry for:

- cited-only references;
- alphabetization;
- author completeness;
- publication status;
- DOI links where available;
- journal/book title formatting;
- duplicates;
- malformed/arXiv metadata.

Do not invent page numbers.

Create:

`manuscript/machine_learning/MLJ_REFERENCE_AUDIT.md`

Report the exact official Springer `.bst`/template option selected and why.

---

## 11. Supplementary Information

Convert the current supplement into Springer Supplementary Information without changing science.

Suggested filename:

`FALI_Machine_Learning_Supplementary_Information.tex`

Preserve:

- Owen derivation/details;
- validation chronology;
- full GPT-2 tables;
- robustness details including failed overall gate;
- gauge tables;
- full Pythia design/results;
- audit terminology where useful.

Use separate SI numbering as appropriate.

Compile to:

`FALI_Machine_Learning_Supplementary_Information.pdf`

---

## 12. Scientific safety checks

Before packaging, verify all of the following remain true:

- L10H2 positive functional recruitment;
- L10H10 attention adaptation without positive recruitment;
- GPT-2 `T>0, R<0`;
- Pythia `T<0, R>0`;
- matched mean/resample preserve tested fine directions;
- overall EF-5 gate failed;
- Pythia mechanism-transfer hypothesis failed;
- gauge claim limited to sign robustness under the prespecified zero-mean sensitivity;
- context `C` is coarse, not a localized pathway;
- no first-self-repair / first-conditional-importance / first-QK-OV / new-Owen-math claims.

No new model experiments.

---

## 13. Evidence audit

Create:

`manuscript/machine_learning/MLJ_FINAL_EVIDENCE_AUDIT.md`

Trace every quantitative main-text statement and every quantitative figure value to retained evidence.

Use the existing current-paper results manifest and final evidence audit as starting points.

Require 100% verified numerical claims.

---

## 14. LaTeX upload structure

Springer explicitly warns against subfolders in LaTeX submissions.

Working repository structure may use folders, but final submission ZIP must be flattened into a clean staging directory.

The flattened source package must include all files needed to compile:

- main `.tex`;
- required section `.tex` files if retained;
- bibliography `.bib` and official `.bst` if required;
- official class/style files required by the template;
- `Fig1.pdf`–`Fig5.pdf` (plus EPS if included for editable artwork);
- SI source if the submission system expects it in the same source package;
- no absolute paths;
- no nested `figures/` or `bibliography/` directories.

Run a clean-directory compile from the exact upload ZIP.

---

## 15. Production quality

Require:

- zero unresolved references;
- zero unresolved citations;
- zero missing figures;
- no duplicate labels;
- no stale Neural Networks/Elsevier text;
- no old Highlights/graphical-abstract references;
- no old Fig4 continuation references;
- no meaningful overfull boxes;
- figure lettering readable at compiled size;
- no font/margin/negative-spacing hacks.

Do not optimize to an arbitrary page count.

---

## 16. Final deliverables

Create:

`manuscript/machine_learning/final/`

with:

- `FALI_Machine_Learning_Manuscript.pdf`
- `FALI_Machine_Learning_Supplementary_Information.pdf`
- `FALI_MLJ_Contribution_Information_Sheet.pdf`
- `FALI_Machine_Learning_LaTeX_Source.zip`
- `FALI_Machine_Learning_Submission_Package.zip`
- `README.txt`

The submission package should include the main PDF, Supplementary Information PDF, Contribution Information Sheet PDF, and clean LaTeX source ZIP.

Do not submit.

---

## 17. Reports

Create:

- `MLJ_REQUIREMENTS_VERIFICATION.md`
- `MLJ_REFERENCE_AUDIT.md`
- `MLJ_FINAL_EVIDENCE_AUDIT.md`
- `MLJ_TEXT_CHANGE_AUDIT.md`
- `MLJ_FIGURE_INTEGRATION_REPORT.md`
- `MLJ_FINAL_PRODUCTION_REPORT.md`

`MLJ_TEXT_CHANGE_AUDIT.md` must compare the Machine Learning manuscript against the approved Neural Networks scientific source and categorize changes as:

- RESEARCH-LEAD MLJ ADAPTATION
- SPRINGER TEMPLATE/MECHANICAL
- PRODUCTION COPYEDIT
- SUBSTANTIVE SCIENTIFIC CHANGE

Target: zero unauthorized substantive scientific changes.

---

## 18. Tests / compile

Run the current-paper critical test subset previously used in production.

Compile:

- main manuscript;
- Supplementary Information;
- Contribution Information Sheet;
- exact flattened source ZIP in a fresh temporary directory.

Do not block on known unrelated historical flaky/hanging tests, but document them.

---

## 19. Commit structure

Prefer:

1. `docs: verify Machine Learning submission requirements`
2. `docs: migrate Factorized FALI manuscript to Springer`
3. `chore: integrate research-lead Springer figures`
4. `docs: add MLJ contribution information sheet`
5. `chore: build Machine Learning submission package`

Push branch:

`manuscript/machine-learning-conversion`

Do not merge automatically.

---

## 20. Final response

Return:

1. final HEAD SHA
2. commit list
3. exact Springer template/class version
4. exact bibliography style selected and evidence for selection
5. main compile result/pages
6. SI compile result/pages
7. Contribution Information Sheet pages
8. abstract word count
9. keyword count
10. unresolved refs/citations
11. overfull-box count
12. figure hash verification result for Fig1–Fig5
13. whether all figures use the research-lead supplied assets unchanged
14. bibliography warnings
15. numerical claims verified count
16. unsupported claims found
17. any substantive scientific text change made by editor
18. critical tests
19. clean flattened-ZIP recompilation result
20. stale Elsevier/Neural Networks strings found
21. exact final deliverable paths
22. author decisions still required: ORCID/postal metadata, additional prior-publication disclosure, final data/code DOI/URL

## STOP

Do not submit.
Do not merge automatically.
Do not rerun model experiments.
Do not independently redraw figures.
Do not alter scientific conclusions.
