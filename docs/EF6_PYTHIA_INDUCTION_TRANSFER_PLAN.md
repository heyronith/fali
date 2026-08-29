# EF-6 — Pythia-160M Repeated-Token Induction Transfer (frozen before execution)

This document preregisters Extended FALI Phase EF-6: the **one authorized**
cross-setting scientific generalization experiment (IOI → induction; GPT-2 → Pythia-160M).

**Preregistration commit SHA** must appear in every EF-6 `run_config.json`.

EF-5 remains formally **EF5_ABLATION_GATE = FAIL**. This preregistration does not
rewrite that historical verdict.

---

## 0. Scientific question

When a causally important induction head in Pythia-160M is removed, can a later
induction head become more causally useful, and if so, does its Extended-FALI Owen
decomposition show **phi_T > 0** and **phi_R < 0** under the same intervention-defined
five-player game used in GPT-2 IOI?

**Not claimed:** universal mechanism, all self-repair, unique natural decomposition,
fully independent architecture family.

---

## 1. Threat model and mitigations

| Threat | Mitigation |
|--------|------------|
| Post-hoc head selection | Frozen algorithm; no phi_* in discovery |
| Multiple models/tasks | Pythia-160m + induction only; DistilGPT2 emergency-only |
| Threshold tuning | 0.05 attention threshold frozen |
| Pair override | selection_lock.json committed before confirmation |

Discovery must never compute, print, serialize, inspect, or rank on phi_T/R/O/V/C or Phi_SCORE.

---

## 2. Model

| Parameter | Value |
|-----------|-------|
| Repository | `EleutherAI/pythia-160m` |
| Checkpoint | Final released |
| Loader | `HookedTransformer.from_pretrained(..., center_writing_weights=True, dtype=float32)` |
| Expected cfg | n_layers=12, n_heads=12, d_model=768 |
| Revision | Resolved once → `model_revision.json`; pinned for all runs |

If revision changes between smoke/discovery/confirmation: **STOP**.

---

## 3. Hardware / numerics

| Parameter | Value |
|-----------|-------|
| Platform | Modal NVIDIA L4 |
| dtype | torch.float32 |
| Mixed precision | Forbidden |
| Quantization | Forbidden |

---

## 4. Synthetic induction dataset

| Parameter | Value |
|-----------|-------|
| Token pool | IDs in `[1000, 10000)` |
| Base length L | 32 |
| Query offsets J | 8, 16, 24 |
| Sampling | 64 jointly unique IDs without replacement per example |
| x block | first 32 IDs; y block | second 32; disjoint |
| BOS | Not prepended |
| Padding | token ID 0 after final query q only |

**Repeated input:** `[x_0..x_31, x_0..x_j]` → length 41/49/57  
**Control input:** `[y_0..y_31, x_0..x_j]` (same length)  
**Target:** `x_{j+1}`  
**Final query:** `q = 32 + j`  
**Induction key:** `k = j + 1`  
**valid_length:** `q + 1`

Target-edge definition frozen: coordinate `(q, k)`.

---

## 5. Discovery / confirmation split

| Phase | Seeds | Examples |
|-------|-------|----------|
| Discovery | 42 only | 128 per offset × 3 = **384** |
| Confirmation | 1, 8, 22 | 128 per offset × 3 per seed = **384/seed** |

Discovery seed 42 never enters confirmatory statistics.  
Confirmation seeds never influence pair selection.

---

## 6. Task metric

```
metric = log_softmax(logits[q])[target_token]
```

Natural-log probability units. No foil tokens.

**induction_advantage** = metric(REPEATED) − metric(CONTROL)

---

## 7. Discovery estimators

**Observed balanced mean:** equal mean within each offset (128 prompts), then equal
average across offsets 8, 16, 24.

**Bootstrap:** 10,000 deterministic SHA256-seeded draws; independent resample-within-offset;
offsets weighted equally.

---

## 8. G0 — Discovery behavior gate

Before head discovery:

- 95% bootstrap CI of induction_advantage entirely > 0
- All three offset means > 0

Failure → **EF6_GATE = FAIL**, STOP. No offset/model/threshold changes.

---

## 9. Engineering smoke (before scientific discovery)

| Parameter | Value |
|-----------|-------|
| Seed | 42 |
| Examples | 8 deterministic (all offsets) |
| Technical primary | L0H0 |
| Technical candidate | L1H0 |

Thresholds: score→pattern ≤2e−6; z ≤1e−5; endpoints ≤1e−4; Owen ≤1e−8;
quotient ≤1e−8; injection-scope ≤1e−7; live-leak Q/V/QV ≤1e−5; partition exact.

Smoke attribution signs must not enter selection.

---

## 10. Score partition (T/R/O)

Generic API: `final_query=q`, `target_key=k`, `valid_length=q+1`.

- **T:** exactly `(q, k)`
- **R:** other valid keys in final query row q
- **O:** valid score coords with query < q

Require T∪R∪O = valid causal scores; pairwise disjoint; exactly one T per example.

---

## 11. Head effect definitions

```
intact_effect_h = metric(clean) − metric(clean with h zeroed)
conditional_effect_c|p = metric(p zeroed) − metric(p and c zeroed)
conditional_gain_c|p = conditional_effect_c|p − intact_effect_c
```

Primary must satisfy `primary.layer < candidate.layer`.

---

## 12. Primary eligibility (discovery)

A head p is eligible iff:

A. clean induction_target_attention ≥ 0.05  
B. intact_effect_p bootstrap CI entirely > 0  
C. all 3 offset intact_effect means > 0  
D. ∃ later-layer head with clean induction_target_attention ≥ 0.05  

---

## 13. Candidate eligibility (discovery)

For eligible primary p, candidate c with p.layer < c.layer and clean attention_c ≥ 0.05:

A. conditional_effect bootstrap CI entirely > 0  
B. all 3 offset conditional_effect means > 0  
C. conditional_gain bootstrap CI entirely > 0  
D. all 3 offset conditional_gain means > 0  

No phi_* allowed during discovery.

---

## 14. Pair selection (frozen)

Among all eligible ordered pairs, maximize exact balanced **conditional_gain**.

Tie-break order:

1. larger conditional_effect  
2. larger candidate clean induction attention  
3. larger primary intact_effect  
4. lower primary layer  
5. lower primary head  
6. lower candidate layer  
7. lower candidate head  

If no eligible pair: **EF6_GATE = FAIL**, valid negative result.

---

## 15. Extended-FALI confirmation game

Five players: T, R, O, V, C. Standard Owen over `{T,R,O}|{V}|{C}`.

32 cells per confirmation example. Utility:

```
U(state) = log P(target | hybrid candidate z, context C)
         − log P(target | candidate zeroed, context C)
```

Require U00000 = intact_effect; U11111 = conditional_effect; Owen efficiency ≤1e−8.

---

## 16. Confirmation estimators

**Observed:** per seed, equal mean across offsets 8/16/24; then equal mean across seeds 1,8,22.

**Bootstrap:** 10,000 draws; resample within seed×offset; offsets equal within seed; seeds equal;
seeds **not** resampled.

Point estimate = observed balanced mean, **not** bootstrap mean.

---

## 17. Frozen confirmatory hypotheses

| Gate | Criterion |
|------|-----------|
| **G0** | held-out induction_advantage > 0 (CI, 3/3 seeds, 3 offsets) |
| **G1** | primary_drop > 0 (CI, 3/3 seeds, 3 offsets) |
| **G2** | conditional_effect > 0 AND conditional_gain > 0 (both full rules) |
| **G3** | phi_T > 0 + \|phi_T\|/score_abs_total ≥ 0.10 |
| **G4** | phi_R < 0 + \|phi_R\|/score_abs_total ≥ 0.10 |

phi_O: descriptive only; not confirmatory.

---

## 18. EF6 gate

| Status | Rule |
|--------|------|
| **PASS** | G0, G1, G2, G3, G4 all CONFIRMED |
| **PARTIAL** | G0+G1+G2 CONFIRMED and exactly one of G3/G4 CONFIRMED |
| **FAIL** | otherwise |

No alternate pair/model after FAIL.

---

## 19. Run order

1. Preregistration commit  
2. Implementation + unit tests (0 failures)  
3. Pythia engineering smoke  
4. Discovery G0 + head scoring + pair selection  
5. Commit selection_lock.json  
6. Confirmation seeds 1, 8, 22  
7. Aggregate analysis  

---

## 20. Outputs

```
results/extended_fali/ef6_pythia_induction/
├── model_revision.json
├── smoke/
├── discovery/
├── confirmation/seed_{1,8,22}/
└── aggregate/
```

Raw audit branch: `audit/extended-fali-ef6-raw` (private).

---

## 21. Historical integrity

EF-6 must not modify EF-4/EF-5/EF-5R artifacts. Non-tautological audit at run time.

---

## 22. Negative result policy

Discovery failure (G0, no primary, no pair) or confirmation G2 failure → FAIL.
No Pythia-410M, DistilGPT2 scientific fallback, or threshold changes without research lead.

---

## 23. Allowed claim if PASS

Held-out Pythia-160M induction: upstream removal increased later-head causal usefulness
with phi_T > 0 and phi_R < 0 under the preregistered Extended-FALI game.

**Not allowed:** universal self-repair, rival inhibition proven, cross-architecture universality.
