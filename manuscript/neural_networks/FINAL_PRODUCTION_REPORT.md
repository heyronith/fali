# FINAL PRODUCTION REPORT — Neural Networks

**Branch:** `manuscript/neural-networks-production`  
**Editorial source HEAD:** `563cc688242a5b04ec9fc857f30e770604f2c743`  
**Date:** 2026-09-15

## Baseline compile (before production edits)

| Item | Value |
|---|---|
| Main | SUCCESS — **28 pages** |
| Supplement | SUCCESS — **7 pages** |
| Unresolved citations | 0 |
| Unresolved references | 0 |
| Duplicate labels | 0 |
| Overfull `\hbox` (main) | **13** |
| Underfull `\hbox` (main) | 0 |
| BibTeX empty-pages warnings | 7 |

### Baseline overfull drivers (main)

- Long unbreakable `SCORE/VALUE/CONTEXT` strings
- Limitations paragraph (endpoints / coalition)
- Appendix path `supplement/` in typewriter
- Minor hyphenation tightness (<5 pt) in related work / AI disclosure

### Baseline figure format

All inclusions use PNG (`Fig1.png` … `Fig5.png`); no PDF vectors in tree at baseline.

### Title mismatch (to fix)

| Location | Title |
|---|---|
| `main.tex` | FALI: Mechanistic Decomposition of Conditional Attention-Head **Causal Usefulness** |
| `supplement/supplement.tex` | … Conditional Attention-Head **Importance** (stale) |

## Production goals

1. Sync supplement title to approved manuscript title.
2. Prefer vector PDF figures from retained plotting scripts without re-running models.
3. Reduce meaningful overfull boxes without margin/font hacks.
4. Clean bibliography empty-pages warnings where metadata allows.
5. Soften main-figure captions that still use internal EF/G gate jargon (science unchanged).
6. Package final deliverables under `final/`.

Updates after production edits are appended below.

---

## Post-production compile (filled after final build)

| Item | Value |
|---|---|
| Main | TBD |
| Supplement | TBD |
| Overfull body | TBD |
| Vector figures | TBD |
| Clean ZIP recompile | TBD |
