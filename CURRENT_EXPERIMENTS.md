# CURRENT EXPERIMENTS

Only experiments needed to understand/reproduce the current Factorized FALI paper.

| experiment | script | purpose | model/task | seeds | outputs | manuscript claim | status |
|---|---|---|---|---|---|---|---|
| EF-0R math | `tests/test_extended_fali_math.py` | Owen correctness | n/a | n/a | tests | Method | PASS |
| EF-1R synthetic | `experiments/extended_fali_ef1_synthetic.py` | factor isolation + TL invariants | synthetic / random TL | 4242 | `results/extended_fali/ef1_synthetic/` | Method Validation | PASS |
| EF-2 smoke | `experiments/extended_fali_ef2_gpt2_smoke.py` | GPT-2 pipeline smoke | GPT-2 IOI | smoke | `ef2_gpt2_smoke/` | provenance | PASS |
| EF-3 two-template | `extended_fali_ef3_*.py` | pre-canonical screen | GPT-2 IOI | 4 seeds | `ef3_two_template/` | GPT-2 narrative | complete |
| EF-4 canonical | `extended_fali_ef4_canonical.py` + analyze | confirmatory fine channels | GPT-2 IOI | 42,1,8,22 | `ef4_canonical/` | GPT-2 $T>0,R<0$ | PASS |
| EF-5 robustness | `extended_fali_ef5_*.py` | matched ablations | GPT-2 IOI | 4 seeds | `ef5_ablation_robustness/` | limited robustness; gate FAIL | FAIL overall / ROBUST channels |
| EF-6 Pythia | `extended_fali_ef6_*.py` | transfer test | Pythia-160M induction | discovery 42; confirm 1,8,22 | `ef6_pythia_induction/` | $T<0,R>0$; transfer FAIL | FAIL transfer |
| Gauge sensitivity | Modal gauge drivers + analyze | row-mean sensitivity | GPT-2 + Pythia | EF-4/EF-6 seeds | `gauge_sensitivity/` | sign robustness claim A | PRESERVED |
| Phase-1 corrected gain | `revision_phase1_*.py` | recruitment vs attention | GPT-2 IOI | 4 seeds | `revision_phase1/` | functional dissociation | complete |
| Phase-4 factorial (figures) | `revision_phase4_*.py` | figure/supplement support | GPT-2 | — | `revision_phase4/` | supplement pointer | complete |

## Archived / deleted (not current)

See `DELETION_MANIFEST.md` for Frozen Attention scripts (`qk_decomposition`, `output_mediation`, historical `fali_decomposition`, review packages, TMLR clutter).
